
#include "Private/Simulation/Traffic/Path/DeepDriveBasePath.h"
#include "Private/Simulation/Traffic/Path/DeepDrivePartialPath.h"
#include "Simulation/Traffic/Path/DeepDrivePathBuilder.h"
#include "Private/Utils/DeepDriveUtils.h"
#include "Private/Simulation/Traffic/Path/Annotations/DeepDrivePathDistanceAnnotation.h"

DEFINE_LOG_CATEGORY(LogDeepDriveBasePath);

DeepDriveBasePath::DeepDriveBasePath(ADeepDriveAgent &agent, const SDeepDriveRoadNetwork &roadNetwork,  UBezierCurveComponent &bezierCmp, const SDeepDrivePathConfiguration &pathCfg)
	:	m_Agent(agent)
	,	m_RoadNetwork(roadNetwork)
	,	m_BezierCurve(bezierCmp)
	,	m_PathConfiguration(pathCfg)
{
}

DeepDriveBasePath::~DeepDriveBasePath()
{
	delete m_PartialPath;
}

void DeepDriveBasePath::setRoute(const SDeepDriveRoute &route)
{
	m_Route = route;
	if(m_PartialPath)
	{
		delete m_PartialPath;
		m_PartialPath = 0;
	}
	convertRouteLinks();
}

void DeepDriveBasePath::advance(float deltaSeconds, float &speed, float &steering, float &brake)
{
	if (m_PartialPath == 0 || m_PartialPath->update() == false)
	{
		// create new partial path
		DeepDrivePartialPath *newPartialPath = createPartialPath();
		m_PartialPath = newPartialPath;
		m_PartialPath->update();
	}

	if (m_PartialPath)
	{
		m_PartialPath->advance(deltaSeconds, speed, steering, brake);
	}
}

bool DeepDriveBasePath::hasReachedDestination() const
{
    return m_PartialPath ? m_PartialPath->hasReachedDestination() : true;
}

float DeepDriveBasePath::getRouteLength() const
{
    return m_PartialPath ? m_PartialPath->getRouteLength() : 0.0f;
}

float DeepDriveBasePath::getDistanceAlongRoute() const
{
    return m_PartialPath ? m_PartialPath->getDistanceAlongRoute() : 0.0f;
}

float DeepDriveBasePath::getDistanceToCenterOfTrack() const
{
    return m_PartialPath ? m_PartialPath->getDistanceToCenterOfTrack() : 0.0f;
}

void DeepDriveBasePath::showPath(UWorld *world)
{
    if(m_PartialPath)
        m_PartialPath->showPath(world);
}

void DeepDriveBasePath::predictPath(DeepDrivePredictedPath &predictedPath, float predictionLength, float curVelocity)
{
    if(m_PartialPath)
    {
        m_PartialPath->predictPath(predictedPath, predictionLength, curVelocity);
    }
}

void DeepDriveBasePath::convertRouteLinks()
{
	m_RouteSegments.Empty();
	int32 curManeuverInd = 0;
	for (signed i = 0; i < m_Route.Links.Num(); ++i)
	{
		const SDeepDriveRoadLink &link = m_RoadNetwork.Links[m_Route.Links[i]];
		const bool lastLink = (i + 1) == m_Route.Links.Num();

		uint32 curLane = link.getRightMostLane(EDeepDriveLaneType::MAJOR_LANE);

		SDeepDriveBasePathSegment basePathSegment;

		const SDeepDriveLane &lane = link.Lanes[curLane];
		basePathSegment.Segments.Append(lane.Segments);

		if (lastLink)
		{
			if (curManeuverInd < m_Route.Maneuvers.Num())
				basePathSegment.Maneuver = m_Route.Maneuvers[curManeuverInd++];
		}
		else
		{
			const SDeepDriveRoadLink &nextLink = m_RoadNetwork.Links[m_Route.Links[i + 1]];
			const SDeepDriveRoadSegment &segment = m_RoadNetwork.Segments[lane.Segments[lane.Segments.Num() - 1]];
			curLane = nextLink.getRightMostLane(EDeepDriveLaneType::MAJOR_LANE);
			const SDeepDriveJunction &junction = m_RoadNetwork.Junctions[link.ToJunctionId];
			junction.findJunctionConnection(m_Route.Links[i], segment.SegmentId, nextLink.Lanes[curLane].Segments[0], basePathSegment.Connection);

			if(curManeuverInd < m_Route.Maneuvers.Num())
			{
				basePathSegment.Maneuver = m_Route.Maneuvers[curManeuverInd++];
				extractCrossTrafficRoads(basePathSegment.Maneuver, junction);
			}
		}

		m_RouteSegments.Add(basePathSegment);
	}
}

void DeepDriveBasePath::extractCrossTrafficRoads(SDeepDriveManeuver &maneuver, const SDeepDriveJunction &junction)
{
	//	CrossTraffic: for each cross traffic road build all cross traffic path
	//	based on each valid from link - to link - connection segment combination

	const FVector2D manFromPoint(m_RoadNetwork.Links[maneuver.FromLinkId].EndPoint);
	const FVector2D manToPoint(m_RoadNetwork.Links[maneuver.ToLinkId].StartPoint);

	FBox2D manArea(manFromPoint, manFromPoint);
	deepdrive::utils::expandBox2D(manArea, manToPoint);

	for(auto &crossTrafficRoad : maneuver.CrossTrafficRoads)
	{
		UE_LOG(LogDeepDriveBasePath, Log, TEXT("Extracting cross traffic roads for junction %d from link %s to link %s"), junction.JunctionId, *(m_RoadNetwork.getDebugLinkName(crossTrafficRoad.FromLinkId)), *(m_RoadNetwork.getDebugLinkName(crossTrafficRoad.ToLinkId)) );
	
		crossTrafficRoad.Paths.Add(TDeepDrivePathPoints());

		TDeepDrivePathPoints &pathPoints = crossTrafficRoad.Paths.Last();

		const SDeepDriveRoadLink &fromLink = m_RoadNetwork.Links[crossTrafficRoad.FromLinkId];
		const SDeepDriveRoadLink &toLink = m_RoadNetwork.Links[crossTrafficRoad.ToLinkId];

		const int32 fromLaneInd = fromLink.getRightMostLane(EDeepDriveLaneType::MAJOR_LANE);
		const int32 toLaneInd = toLink.getRightMostLane(EDeepDriveLaneType::MAJOR_LANE);

		if(fromLaneInd >= 0 && toLaneInd >= 0)
		{
			const SDeepDriveRoadSegment &fromSegment = m_RoadNetwork.Segments[fromLink.Lanes[fromLaneInd].Segments.Last()];
			const SDeepDriveRoadSegment &toSegment = m_RoadNetwork.Segments[toLink.Lanes[toLaneInd].Segments[0]];
			SDeepDriveJunctionConnection junctionConnection;
			if (junction.findJunctionConnection(crossTrafficRoad.FromLinkId, fromSegment.SegmentId, toSegment.SegmentId, junctionConnection))
			{
				DeepDrivePathBuilder pathBuilder(m_RoadNetwork, pathPoints, m_BezierCurve);
				pathBuilder.buildPath(fromSegment, toSegment, m_RoadNetwork.Segments[junctionConnection.ConnectionSegmentId], crossTrafficRoad.FromLength, crossTrafficRoad.ToLength);

				DeepDrivePathDistanceAnnotation distanceAnnotation;
				distanceAnnotation.annotate(pathPoints, 0.0f);

				FBox2D curManArea = pathBuilder.getArea();
				deepdrive::utils::expandBox2D(manArea, curManArea.Min);
				deepdrive::utils::expandBox2D(manArea, curManArea.Max);
				UE_LOG(LogDeepDriveBasePath, Log, TEXT("Extracted path with %d points fromLength %f toLength %f ManArea %s"), crossTrafficRoad.Paths.Last().Num(), crossTrafficRoad.FromLength, crossTrafficRoad.ToLength, *(curManArea.ToString()));
			}
		}
	}
	manArea.ExpandBy(1.1f);
	maneuver.ManeuverArea = manArea;
	UE_LOG(LogDeepDriveBasePath, Log, TEXT("Total Maneuver Area %s"), *(maneuver.ManeuverArea.ToString()));
}

DeepDrivePartialPath* DeepDriveBasePath::createPartialPath()
{
	DeepDrivePartialPath *path = new DeepDrivePartialPath(m_Agent, m_RoadNetwork, m_BezierCurve, m_PathConfiguration);

	path->setup(m_RouteSegments);
	path->trimStart(m_Route.Start);
	path->trimEnd(m_Route.Destination);
	path->annotate();

	return path;
}
