

#include "DeepDrivePluginPrivatePCH.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoute.h"
#include "Runtime/Engine/Classes/Components/SplineComponent.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"

DEFINE_LOG_CATEGORY(LogDeepDriveRoute);

// Sets default values
ADeepDriveRoute::ADeepDriveRoute()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;
}

void ADeepDriveRoute::Tick(float DeltaTime)
{
	if(m_RoutePoints.Num() > 0)
	{
		FColor col = FColor::Green;
		const uint8 prio = 40;

		DrawDebugPoint(GetWorld(), m_RoutePoints[0].Location, 10.0f, col, false, 0.0f, prio);
		for(signed i = 1; i < m_RoutePoints.Num(); ++i)
		{
			DrawDebugPoint(GetWorld(), m_RoutePoints[i].Location, 10.0f, col, false, 0.0f, prio);
			DrawDebugLine(GetWorld(), m_RoutePoints[i - 1].Location, m_RoutePoints[i].Location, col, false, 0.0f, prio, 4.0f);
		}
	}
}

void ADeepDriveRoute::initialize(const SDeepDriveRoadNetwork &roadNetwork, const SDeepDriveRouteData &routeData)
{
	m_RoadNetwork = &roadNetwork;
	m_RouteData = routeData;
}

void ADeepDriveRoute::convert(const FVector &location)
{
	convertToPoints(location);
	if (m_RoutePoints.Num() > 0)
	{
		annotateRoute();
	}
	UE_LOG(LogDeepDriveRoute, Log, TEXT("Route converted numPoints %d length %f"), m_RoutePoints.Num(), m_RouteLength);
}

void ADeepDriveRoute::convertToPoints(const FVector &location)
{
	m_RouteLength = 0.0f;
	m_RoutePoints.Empty();
	if	(	m_RoadNetwork
		&&	m_RouteData.Links.Num() > 0
		)
	{
		float carryOverDistance = 0.0f;
		uint32 curLane = m_RoadNetwork->Links[m_RouteData.Links[0]].getRightMostLane(EDeepDriveLaneType::MAJOR_LANE);
		for (signed i = 0; i < m_RouteData.Links.Num(); ++i)
		{
			const SDeepDriveRoadLink &link = m_RoadNetwork->Links[m_RouteData.Links[i]];
			const bool lastLink = (i + 1) == m_RouteData.Links.Num();

			const SDeepDriveLane &lane = link.Lanes[curLane];

			for (signed j = 0; j < lane.Segments.Num(); ++j)
			{
				const SDeepDriveRoadSegment &segment = m_RoadNetwork->Segments[lane.Segments[j]];
				const bool lastSegment = (j + 1) == lane.Segments.Num();
				carryOverDistance = addSegmentToPoints(segment, lastLink && lastSegment, carryOverDistance);
			}

			if (!lastLink)
			{
				const SDeepDriveRoadLink &nextLink = m_RoadNetwork->Links[m_RouteData.Links[i + 1]];
				const SDeepDriveRoadSegment &segment = m_RoadNetwork->Segments[lane.Segments[lane.Segments.Num() - 1]];

				curLane = nextLink.getRightMostLane(EDeepDriveLaneType::MAJOR_LANE);

				const SDeepDriveJunction &junction = m_RoadNetwork->Junctions[link.ToJunctionId];
				const uint32 connectionSegmentId = junction.findConnectionSegment(segment.SegmentId, nextLink.Lanes[curLane].Segments[0]);
				if (connectionSegmentId)
					carryOverDistance = addSegmentToPoints(m_RoadNetwork->Segments[connectionSegmentId], false, carryOverDistance);
			}
		}
	}
}

void ADeepDriveRoute::annotateRoute()
{
	const int32 numPoints = m_RoutePoints.Num();

	// calculate route length
	FVector last = m_RoutePoints[0].Location;
	m_RoutePoints[0].RemainingDistance = 0.0f;
	for (signed i = 1; i < numPoints; ++i)
	{
		const FVector &cur = m_RoutePoints[i].Location;
		m_RouteLength += (cur - last).Size();
		last = cur;
		m_RoutePoints[i].RemainingDistance = m_RouteLength;
	}

	// correct remaining distance and setup speed limits
	float speedLimit = DeepDriveRoadNetwork::SpeedLimitInTown;
	uint32 curSegmentId = 0;
	for (int32 i = 0; i < numPoints; ++i)
	{
		SRoutePoint &rp = m_RoutePoints[i];

		//	correct remaining distance
		rp.RemainingDistance = m_RouteLength - rp.RemainingDistance;

		if(rp.SegmentId != curSegmentId)
		{
			curSegmentId = rp.SegmentId;
			const SDeepDriveRoadSegment &segment = m_RoadNetwork->Segments[curSegmentId];
			speedLimit = segment.SpeedLimit;
			if(segment.IsConnection && segment.SlowDownDistance > 0.0f)
			{
				float coveredDist = 0.0f;
				FVector lastPoint = m_RoutePoints[i].Location;
				for(int32 j = i - 1; j >= 0; --j)
				{
					const FVector curPoint = m_RoutePoints[j].Location;
					coveredDist += (curPoint - lastPoint).Size();
					if (coveredDist < segment.SlowDownDistance)
					{
						m_RoutePoints[j].SpeedLimit = speedLimit;
						lastPoint = curPoint;
					}
					else
						break;
				}
			}
		}

		rp.SpeedLimit = speedLimit;
	}
}

float ADeepDriveRoute::addSegmentToPoints(const SDeepDriveRoadSegment &segment, bool addEnd, float carryOverDistance)
{
	float curDist = carryOverDistance;
	float segmentLength = 0.0f;
	if (segment.SplinePoints.Num() > 0)
	{
		segmentLength = segment.SplineCurves.GetSplineLength();
		while (curDist < segmentLength)
		{
			SRoutePoint rp;
			rp.SegmentId = segment.SegmentId;
			rp.RelativePosition = curDist / segmentLength;
			const float key = segment.SplineCurves.ReparamTable.Eval(curDist, 0.0f);
			rp.Location = segment.Transform.TransformPosition(segment.SplineCurves.Position.Eval(key, FVector::ZeroVector));

			m_RoutePoints.Add(rp);

			curDist += m_StepSize;
		}
	}
	else
	{
		const FVector dir = segment.EndPoint - segment.StartPoint;
		segmentLength = dir.Size();
		while (curDist < segmentLength)
		{
			SRoutePoint rp;
			rp.SegmentId = segment.SegmentId;
			rp.RelativePosition = curDist / segmentLength;
			rp.Location = rp.RelativePosition * dir + segment.StartPoint;
			m_RoutePoints.Add(rp);

			curDist += m_StepSize;
		}
	}
	carryOverDistance = curDist - segmentLength;
	return carryOverDistance;
}

void ADeepDriveRoute::update(ADeepDriveAgent &agent)
{
	findClosestRoutePoint(agent);

}

float ADeepDriveRoute::getRemainingDistance()
{
	return		m_curRoutePointIndex >= 0 && m_curRoutePointIndex < m_RoutePoints.Num()
			?	m_RoutePoints[m_curRoutePointIndex].RemainingDistance
			: 	0.0f;
}

FVector ADeepDriveRoute::getLocationAhead(float distanceAhead, float sideOffset)
{
	int32 ind = m_curRoutePointIndex + static_cast<int32> (distanceAhead / m_StepSize);
	FVector locAhead = m_RoutePoints[ind % m_RoutePoints.Num()].Location;
	return locAhead;
}

float ADeepDriveRoute::getSpeedLimit(float distanceAhead)
{
	float speedLimit = 0.0f;
	if(m_curRoutePointIndex >= 0)
	{
		if(distanceAhead > 0.0f)
		{
			const int32 index = getPointIndexAhead(distanceAhead);
			if(index >= 0)
				speedLimit = m_RoutePoints[index].SpeedLimit;
		}
		else
			speedLimit = m_RoutePoints[m_curRoutePointIndex % m_RoutePoints.Num()].SpeedLimit;
	}

	return speedLimit;
}

void ADeepDriveRoute::findClosestRoutePoint(ADeepDriveAgent &agent)
{
	FVector agentPos = agent.GetActorLocation();
	float bestDist = 1000000.0f;
	m_curRoutePointIndex = -1;
	for(signed i = m_RoutePoints.Num() - 1; i >= 0; --i)
	{
		const float curDist = (agentPos - m_RoutePoints[i].Location).Size();
		if (curDist <= bestDist)
		{
			bestDist = curDist;
			m_curRoutePointIndex = i;
		}
	}
}

int32 ADeepDriveRoute::getPointIndexAhead(float distanceAhead)
{
	int32 curIndex = m_curRoutePointIndex;
	FVector lastPoint = m_RoutePoints[curIndex].Location;
	for( ; curIndex < m_RoutePoints.Num() && distanceAhead > 0.0f; ++curIndex)
	{
		FVector curPoint = m_RoutePoints[curIndex].Location;
		distanceAhead -= (curPoint - lastPoint).Size();
		lastPoint = curPoint;
	}

	return curIndex < m_RoutePoints.Num() ? curIndex : m_RoutePoints.Num() - 1;
}
