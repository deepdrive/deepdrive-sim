

#include "Public/Simulation/RoadNetwork/DeepDriveRoadNetworkExtractor.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoadNetwork.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoadSegmentProxy.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoadLinkProxy.h"
#include "Public/Simulation/RoadNetwork/DeepDriveJunctionProxy.h"

DeepDriveRoadNetworkExtractor::DeepDriveRoadNetworkExtractor(UWorld *world)
	:	m_World(world)
{

}

/**
 * 	Extract road network by iterating over all junctions.
 * 	Add each referenced link and each segment referenced by this link.
 * 	If a link is empty a segment equal to that link is automatically generated and added as a single lane for that link.
 */

void DeepDriveRoadNetworkExtractor::extract(SDeepDriveRoadNetwork &roadNetwork)
{
	TArray<AActor *> junctions;
	UGameplayStatics::GetAllActorsOfClass(m_World, ADeepDriveJunctionProxy::StaticClass(), junctions);
	for (auto &actor : junctions)
	{
		ADeepDriveJunctionProxy *junctionProxy = Cast<ADeepDriveJunctionProxy>(actor);
		if (junctionProxy)
		{
			uint32 junctionId = m_nextJunctionId++;

			SDeepDriveJunction junction;
			junction.JunctionId = junctionId;

			junction.Center = FVector::ZeroVector;
			int32 count = 0;
			for (auto &linkProxy : junctionProxy->getLinksIn())
			{
				const uint32 linkId = addLink(roadNetwork, *linkProxy);
				if (linkId)
				{
					junction.LinksIn.Add(linkId);
					roadNetwork.Links[linkId].ToJunctionId = junctionId;
					junction.Center += roadNetwork.Links[linkId].StartPoint;
					count++;
				}
			}

			for (auto &linkProxy : junctionProxy->getLinksOut())
			{
				const uint32 linkId = addLink(roadNetwork, *linkProxy);
				if (linkId)
				{
					junction.LinksOut.Add(linkId);
					roadNetwork.Links[linkId].FromJunctionId = junctionId;
					junction.Center += roadNetwork.Links[linkId].EndPoint;
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
					connection.ConnectionSegment 	=	connectionProxy.ConnectionSegment
													?	addSegment(roadNetwork, *connectionProxy.ConnectionSegment, 0)
													:	addConnectionSegment(roadNetwork, connection.FromSegment, connection.ToSegment);
				}

				junction.Connections.Add(connection);
			}

			roadNetwork.Junctions.Add(junctionId, junction);
		}
	}
}

uint32 DeepDriveRoadNetworkExtractor::addConnectionSegment(SDeepDriveRoadNetwork &roadNetwork, uint32 fromSegment, uint32 toSegment)
{
	uint32 connectionId = m_nextSegmentId++;

	SDeepDriveRoadSegment segment;
	segment.SegmentId = connectionId;
	segment.LinkId = 0;
	segment.StartPoint = roadNetwork.Segments[fromSegment].EndPoint;
	segment.EndPoint = roadNetwork.Segments[toSegment].StartPoint;
	segment.Heading = calcHeading(segment.StartPoint, segment.EndPoint);

	segment.SpeedLimit = roadNetwork.Segments[fromSegment].SpeedLimit;

	segment.IsConnection = true;
	segment.SlowDownDistance = -1;

	roadNetwork.Segments.Add(connectionId, segment);

	return connectionId;
}

uint32 DeepDriveRoadNetworkExtractor::getRoadLink(ADeepDriveRoadLinkProxy *linkProxy)
{
	FString proxyObjName = UKismetSystemLibrary::GetObjectName(linkProxy);
	return m_LinkCache.Contains(proxyObjName) ? m_LinkCache[proxyObjName] : 0;
}

uint32 DeepDriveRoadNetworkExtractor::addLink(SDeepDriveRoadNetwork &roadNetwork, ADeepDriveRoadLinkProxy &linkProxy)
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
					const uint32 segmentId = addSegment(roadNetwork, *segProxy, &link);
					lane.Segments.Add(segmentId);
				}
				link.Lanes.Add(lane);
			}
		}
		else
		{
			SDeepDriveLane lane;
			lane.LaneType = EDeepDriveLaneType::MAJOR_LANE;

			const uint32 segmentId = addSegment(roadNetwork, linkProxy, &link);
			lane.Segments.Add(segmentId);

			link.Lanes.Add(lane);
		}

		m_LinkCache.Add(proxyObjName, linkId);
		roadNetwork.Links.Add(linkId, link);
	}
	else
	{
		linkId = m_LinkCache.FindChecked(proxyObjName);
	}

	return linkId;
}

uint32 DeepDriveRoadNetworkExtractor::addSegment(SDeepDriveRoadNetwork &roadNetwork, ADeepDriveRoadSegmentProxy &segmentProxy, const SDeepDriveRoadLink *link)
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
		roadNetwork.Segments.Add(segmentId, segment);
	}
	else
	{
		segmentId = m_SegmentCache.FindChecked(proxyObjName);
	}

	return segmentId;
}

// add segment based on link proxy
uint32 DeepDriveRoadNetworkExtractor::addSegment(SDeepDriveRoadNetwork &roadNetwork, ADeepDriveRoadLinkProxy &linkProxy, const SDeepDriveRoadLink *link)
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

		segment.SpeedLimit = link ? link->SpeedLimit : DeepDriveRoadNetwork::SpeedLimitConnection;

		segment.IsConnection = false;
		segment.SlowDownDistance = -1.0f;

		m_SegmentCache.Add(proxyObjName, segmentId);
		roadNetwork.Segments.Add(segmentId, segment);
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
