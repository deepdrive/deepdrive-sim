

#include "Public/Simulation/RoadNetwork/DeepDriveRoadNetworkExtractor.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoadNetwork.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoadSegmentProxy.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoadLinkProxy.h"
#include "Public/Simulation/RoadNetwork/DeepDriveJunctionProxy.h"

DEFINE_LOG_CATEGORY(LogDeepDriveRoadNetworkExtractor);

DeepDriveRoadNetworkExtractor::DeepDriveRoadNetworkExtractor(UWorld *world, SDeepDriveRoadNetwork &roadNetwork)
	:	m_World(world)
	,	m_RoadNetwork(roadNetwork)
{

}

/**
 * 	Extract road network by iterating over all junctions.
 * 	Add each referenced link and each segment referenced by this link.
 * 	If a link is empty a segment equal to that link is automatically generated and added as a single lane for that link.
 */

void DeepDriveRoadNetworkExtractor::extract()
{
	TArray<AActor *> junctions;
	UGameplayStatics::GetAllActorsOfClass(m_World, ADeepDriveJunctionProxy::StaticClass(), junctions);
	for (auto &actor : junctions)
	{
		ADeepDriveJunctionProxy *junctionProxy = Cast<ADeepDriveJunctionProxy>(actor);
		if (junctionProxy)
		{
			uint32 junctionId = m_nextJunctionId++;

			UE_LOG(LogDeepDriveRoadNetworkExtractor, Log, TEXT("Extracting junction %d %s"), junctionId, *(UKismetSystemLibrary::GetObjectName(junctionProxy)) );

			SDeepDriveJunction junction;
			junction.JunctionId = junctionId;

			junction.Center = FVector::ZeroVector;
			int32 count = 0;
			for (auto &linkProxy : junctionProxy->getLinksIn())
			{
				const uint32 linkId = addLink(*linkProxy);
				if (linkId)
				{
					junction.LinksIn.Add(linkId);
					m_RoadNetwork.Links[linkId].ToJunctionId = junctionId;
					junction.Center += m_RoadNetwork.Links[linkId].StartPoint;
					count++;
				}
			}

			for (auto &linkProxy : junctionProxy->getLinksOut())
			{
				const uint32 linkId = addLink(*linkProxy);
				if (linkId)
				{
					junction.LinksOut.Add(linkId);
					m_RoadNetwork.Links[linkId].FromJunctionId = junctionId;
					junction.Center += m_RoadNetwork.Links[linkId].EndPoint;
					count++;
				}
			}
			junction.Center /= static_cast<float> (count);

			for (auto &connectionProxy : junctionProxy->getLaneConnections())
			{
				SDeepDriveLaneConnection connection;

				FString fromName = UKismetSystemLibrary::GetObjectName(connectionProxy.FromSegment);
				uint32 fromSegment = m_SegmentCache.Contains(fromName) ? m_SegmentCache[fromName] : 0;
				if(fromSegment == 0)
				{
					fromName = buildSegmentName(UKismetSystemLibrary::GetObjectName(connectionProxy.FromLink));
					fromSegment = m_SegmentCache.Contains(fromName) ? m_SegmentCache[fromName] : 0;
				}

				FString toName = UKismetSystemLibrary::GetObjectName(connectionProxy.ToSegment);
				uint32 toSegment = m_SegmentCache.Contains(toName) ? m_SegmentCache[toName] : 0;
				if(toSegment == 0)
				{
					toName = buildSegmentName(UKismetSystemLibrary::GetObjectName(connectionProxy.ToLink));
					toSegment = m_SegmentCache.Contains(toName) ? m_SegmentCache[toName] : 0;
				}

				if(fromSegment && toSegment)
				{
					connection.FromSegment = fromSegment;
					connection.ToSegment = toSegment;
					if	(	connectionProxy.GenerateAutoConnection
						||	connection.ConnectionSegment == false
						)
					{
						const float speedLimit = connectionProxy.GenerateAutoConnection ? connectionProxy.SpeedLimit : m_RoadNetwork.Segments[fromSegment].SpeedLimit;
						const float slowDownDist = connectionProxy.GenerateAutoConnection ? connectionProxy.SlowDownDistance : -1.0f;
						if(connectionProxy.GenerateAutoConnection && connectionProxy.GenerateCurve)
						{
							connection.ConnectionSegment = addStraightConnectionSegment(connection.FromSegment, connection.ToSegment, speedLimit, slowDownDist, true);
						}
						else
						{
							connection.ConnectionSegment = addStraightConnectionSegment(connection.FromSegment, connection.ToSegment, speedLimit, slowDownDist, false);
						}
					}
					else
					{
						connection.ConnectionSegment = addSegment(*connectionProxy.ConnectionSegment, 0, EDeepDriveLaneType::CONNECTION);
					}
				}
				junction.Connections.Add(connection);

				for(auto &turningRestrictionProxy : junctionProxy->getTurningRestrictions())
				{
					FString fromName = UKismetSystemLibrary::GetObjectName(turningRestrictionProxy.FromLink);
					FString toName = UKismetSystemLibrary::GetObjectName(turningRestrictionProxy.ToLink);
					if	(	m_LinkCache.Contains(fromName)
						&&	m_LinkCache.Contains(toName)
						)
					{
						SDeepDriveTurningRestriction turningRestriction;
						turningRestriction.FromLink = m_LinkCache[fromName];
						turningRestriction.ToLink = m_LinkCache[toName];
						junction.TurningRestrictions.Add(turningRestriction);
					}
				}

			}

			m_RoadNetwork.Junctions.Add(junctionId, junction);
		}
	}
}

uint32 DeepDriveRoadNetworkExtractor::addStraightConnectionSegment(uint32 fromSegment, uint32 toSegment, float speedLimit, float slowDownDistance, bool generateCurve)
{
	uint32 connectionId = m_nextSegmentId++;

	SDeepDriveRoadSegment segment;
	segment.SegmentId = connectionId;
	segment.LinkId = 0;
	segment.StartPoint = m_RoadNetwork.Segments[fromSegment].EndPoint;
	segment.EndPoint = m_RoadNetwork.Segments[toSegment].StartPoint;
	segment.Heading = calcHeading(segment.StartPoint, segment.EndPoint);
	segment.LaneType = EDeepDriveLaneType::CONNECTION;

	segment.SpeedLimit = speedLimit;

	segment.IsConnection = true;
	segment.GenerateCurve = generateCurve;
	segment.SlowDownDistance = slowDownDistance;

	m_RoadNetwork.Segments.Add(connectionId, segment);

	return connectionId;
}

uint32 DeepDriveRoadNetworkExtractor::getRoadLink(ADeepDriveRoadLinkProxy *linkProxy)
{
	FString proxyObjName = UKismetSystemLibrary::GetObjectName(linkProxy);
	return m_LinkCache.Contains(proxyObjName) ? m_LinkCache[proxyObjName] : 0;
}

uint32 DeepDriveRoadNetworkExtractor::addLink(ADeepDriveRoadLinkProxy &linkProxy)
{
	uint32 linkId = 0;
	FString proxyObjName = UKismetSystemLibrary::GetObjectName(&linkProxy);
	if (m_LinkCache.Contains(proxyObjName) == false)
	{
		linkId = m_nextLinkId++;

		SDeepDriveRoadLink link;
		link.LinkId = linkId;
		link.StartPoint = linkProxy.getStartPoint();
		link.EndPoint = linkProxy.getEndPoint();
		link.Heading = calcHeading(link.StartPoint, link.EndPoint);
		link.SpeedLimit = linkProxy.getSpeedLimit();
		link.FromJunctionId = 0;
		link.ToJunctionId = 0;

		const TArray<FDeepDriveLaneProxy> &lanes = linkProxy.getLanes();
		if(lanes.Num() > 0)
		{
			for (auto &laneProxy : lanes)
			{
				SDeepDriveLane lane;
				lane.LaneType = laneProxy.LaneType;

				for (auto &segProxy : laneProxy.Segments)
				{
					const uint32 segmentId = addSegment(*segProxy, &link, lane.LaneType);
					lane.Segments.Add(segmentId);
				}
				link.Lanes.Add(lane);
			}
		}
		else
		{
			SDeepDriveLane lane;
			lane.LaneType = EDeepDriveLaneType::MAJOR_LANE;

			const uint32 segmentId = addSegment(linkProxy, &link);
			lane.Segments.Add(segmentId);

			link.Lanes.Add(lane);
		}

		m_LinkCache.Add(proxyObjName, linkId);
		m_RoadNetwork.Links.Add(linkId, link);

		UE_LOG(LogDeepDriveRoadNetworkExtractor, Log, TEXT("Added link %d %s"), linkId, *(proxyObjName) );

	}
	else
	{
		linkId = m_LinkCache.FindChecked(proxyObjName);
	}

	return linkId;
}

uint32 DeepDriveRoadNetworkExtractor::addSegment(ADeepDriveRoadSegmentProxy &segmentProxy, const SDeepDriveRoadLink *link, EDeepDriveLaneType laneType)
{
	uint32 segmentId = 0;
	FString proxyObjName = UKismetSystemLibrary::GetObjectName(&segmentProxy);
	if (m_SegmentCache.Contains(proxyObjName) == false)
	{
		segmentId = m_nextSegmentId++;

		SDeepDriveRoadSegment segment;
		segment.SegmentId = segmentId;
		segment.LinkId = link ? link->LinkId : 0;
		segment.StartPoint = segmentProxy.getStartPoint();
		segment.EndPoint = segmentProxy.getEndPoint();
		segment.Heading = calcHeading(segment.StartPoint, segment.EndPoint);
		segment.LaneType = laneType;

		const float speedLimit = segmentProxy.getSpeedLimit();
		if (speedLimit <= 0.0f)
			segment.SpeedLimit = link ? link->SpeedLimit : DeepDriveRoadNetwork::SpeedLimitConnection;
		else
			segment.SpeedLimit = speedLimit;

		segment.IsConnection = segmentProxy.isConnection();
		segment.SlowDownDistance = segmentProxy.getSlowDownDistance();

		const USplineComponent *spline = segmentProxy.getSpline();
		if(spline && spline->GetNumberOfSplinePoints() > 2)
		{
			for (signed i = 0; i < spline->GetNumberOfSplinePoints(); ++i)
			{
				FSplinePoint splinePoint;

				splinePoint.Type = spline->GetSplinePointType(i);
				splinePoint.Position = spline->GetLocationAtSplinePoint(i, ESplineCoordinateSpace::World);
				splinePoint.Rotation = spline->GetRotationAtSplinePoint(i, ESplineCoordinateSpace::World);
				splinePoint.Scale = spline->GetScaleAtSplinePoint(i);
				splinePoint.ArriveTangent = spline->GetArriveTangentAtSplinePoint(i, ESplineCoordinateSpace::World);
				splinePoint.LeaveTangent = spline->GetLeaveTangentAtSplinePoint(i, ESplineCoordinateSpace::World);

				segment.SplinePoints.Add(splinePoint);
				segment.Transform = spline->GetComponentTransform();
			}

			segment.SplineCurves = spline->SplineCurves;
		}

		m_SegmentCache.Add(proxyObjName, segmentId);
		m_RoadNetwork.Segments.Add(segmentId, segment);
	}
	else
	{
		segmentId = m_SegmentCache.FindChecked(proxyObjName);
	}

	return segmentId;
}

// add segment based on link proxy
uint32 DeepDriveRoadNetworkExtractor::addSegment(ADeepDriveRoadLinkProxy &linkProxy, const SDeepDriveRoadLink *link)
{
	uint32 segmentId = 0;
	FString proxyObjName = buildSegmentName(UKismetSystemLibrary::GetObjectName(&linkProxy));
	if (m_SegmentCache.Contains(proxyObjName) == false)
	{
		segmentId = m_nextSegmentId++;

		SDeepDriveRoadSegment segment;
		segment.SegmentId = segmentId;
		segment.LinkId = link ? link->LinkId : 0;
		segment.StartPoint = linkProxy.getStartPoint();
		segment.EndPoint = linkProxy.getEndPoint();
		segment.Heading = calcHeading(segment.StartPoint, segment.EndPoint);
		segment.LaneType = EDeepDriveLaneType::MAJOR_LANE;

		segment.SpeedLimit = link ? link->SpeedLimit : DeepDriveRoadNetwork::SpeedLimitConnection;

		segment.IsConnection = false;
		segment.SlowDownDistance = -1.0f;

		m_SegmentCache.Add(proxyObjName, segmentId);
		m_RoadNetwork.Segments.Add(segmentId, segment);
	}
	else
	{
		segmentId = m_SegmentCache.FindChecked(proxyObjName);
	}

	return segmentId;
}

float DeepDriveRoadNetworkExtractor::calcHeading(const FVector &from, const FVector &to)
{
	FVector2D dir = FVector2D(to - from);
	dir.Normalize();
	return FMath::RadiansToDegrees(FMath::Atan2(dir.Y, dir.X));
}
