

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

					switch(connectionProxy.ConnectionShape)
					{
						case EDeepDriveConnectionShape::STRAIGHT_LINE:
						case EDeepDriveConnectionShape::QUADRATIC_SPLINE:
						case EDeepDriveConnectionShape::CUBIC_SPLINE:
						case EDeepDriveConnectionShape::UTURN_SPLINE:
							connection.ConnectionSegment = addConnectionSegment(connection.FromSegment, connection.ToSegment, connectionProxy);
							break;
						case EDeepDriveConnectionShape::ROAD_SEGMENT:
							if(connectionProxy.ConnectionSegment)
								connection.ConnectionSegment = addSegment(*connectionProxy.ConnectionSegment, 0, EDeepDriveLaneType::CONNECTION);
							else
								connection.ConnectionSegment = addStraightConnectionSegment(connection.FromSegment, connection.ToSegment, connectionProxy.SpeedLimit, connectionProxy.SlowDownDistance, false);
							break;
					}
				}
				junction.Connections.Add(connection);

				for(auto &turningRestrictionProxy : junctionProxy->getTurningRestrictions())
				{
					fromName = UKismetSystemLibrary::GetObjectName(turningRestrictionProxy.FromLink);
					toName = UKismetSystemLibrary::GetObjectName(turningRestrictionProxy.ToLink);
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

	for(auto &linkProxy : m_LinkProxies)
	{
		FString oppositeName = linkProxy->getOppositeDirection() ? UKismetSystemLibrary::GetObjectName(linkProxy->getOppositeDirection()) : "";
		if(oppositeName != "")
		{
			m_RoadNetwork.Links[m_LinkCache[UKismetSystemLibrary::GetObjectName(linkProxy)]].OppositeDirectionLink = m_LinkCache[oppositeName];
		}
	}




}

uint32 DeepDriveRoadNetworkExtractor::addConnectionSegment(uint32 fromSegment, uint32 toSegment, const FDeepDriveLaneConnectionProxy &connectionProxy)
{
	uint32 connectionId = m_nextSegmentId++;

	SDeepDriveRoadSegment segment;
	segment.SegmentId = connectionId;
	segment.LinkId = 0;
	segment.StartPoint = m_RoadNetwork.Segments[fromSegment].EndPoint;
	segment.EndPoint = m_RoadNetwork.Segments[toSegment].StartPoint;
	segment.Heading = calcHeading(segment.StartPoint, segment.EndPoint);
	segment.LaneType = EDeepDriveLaneType::CONNECTION;

	segment.SpeedLimit = connectionProxy.SpeedLimit;
	segment.ConnectionShape = connectionProxy.ConnectionShape;
	segment.SlowDownDistance = connectionProxy.SlowDownDistance;
	segment.CustomCurveParams[0] = connectionProxy.CustomCurveParams.Parameter0;
	segment.CustomCurveParams[1] = connectionProxy.CustomCurveParams.Parameter1;
	segment.CustomCurveParams[2] = connectionProxy.CustomCurveParams.Parameter2;
	segment.CustomCurveParams[3] = connectionProxy.CustomCurveParams.Parameter3;
	segment.CustomCurveParams[4] = connectionProxy.CustomCurveParams.Parameter4;
	segment.CustomCurveParams[5] = connectionProxy.CustomCurveParams.Parameter5;
	segment.CustomCurveParams[6] = connectionProxy.CustomCurveParams.Parameter6;
	segment.CustomCurveParams[7] = connectionProxy.CustomCurveParams.Parameter7;

	m_RoadNetwork.Segments.Add(connectionId, segment);

	return connectionId;
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
	segment.ConnectionShape = generateCurve ? EDeepDriveConnectionShape::QUADRATIC_SPLINE : EDeepDriveConnectionShape::STRAIGHT_LINE;
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

		m_LinkProxies.Add(&linkProxy);

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

		segment.ConnectionShape = segment.LaneType == EDeepDriveLaneType::CONNECTION ? EDeepDriveConnectionShape::ROAD_SEGMENT : EDeepDriveConnectionShape::NO_CONNECTION;

		const float speedLimit = segmentProxy.getSpeedLimit();
		if (speedLimit <= 0.0f)
			segment.SpeedLimit = link ? link->SpeedLimit : DeepDriveRoadNetwork::SpeedLimitConnection;
		else
			segment.SpeedLimit = speedLimit;

		const USplineComponent *spline = segmentProxy.getSpline();
		if(spline && spline->GetNumberOfSplinePoints() > 2)
		{
			segment.SplineCurves = spline->SplineCurves;
			segment.SplineTransform = spline->GetComponentTransform();
		}

		// segment.Tra nsform = 

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
		segment.ConnectionShape = EDeepDriveConnectionShape::NO_CONNECTION;
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
