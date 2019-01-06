

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

	m_RouteSpline = CreateDefaultSubobject<USplineComponent>(TEXT("RouteSpline"));
	RootComponent = m_RouteSpline;

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
	else
	{
		const float length = m_RouteSpline->GetSplineLength();
		if (length > 110.0f)
		{
			FColor col = FColor::Green;
			const uint8 prio = 40;
			float curDist = 0.0f;
			const float deltaDist = 100.0f;
			DrawDebugPoint(GetWorld(), getSplinePoint(curDist), 10.0f, col, false, 0.0f, prio);
			while (curDist < length)
			{
				float lastDist = curDist;
				curDist += deltaDist;
				DrawDebugPoint(GetWorld(), getSplinePoint(curDist), 10.0f, col, false, 0.0f, prio);
				DrawDebugLine(GetWorld(), getSplinePoint(lastDist), getSplinePoint(curDist), col, false, 0.0f, prio, 4.0f);
			}
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
	// convertToSpline(location);
	convertToPoints(location);
}

void ADeepDriveRoute::convertToPoints(const FVector &location)
{
	m_RoutePoints.Empty();
	if (m_RoadNetwork)
	{
		float carryOverDistance = 0.0f;
		for (signed i = 0; i < m_RouteData.Links.Num(); ++i)
		{
			const SDeepDriveRoadLink &link = m_RoadNetwork->Links[m_RouteData.Links[i]];
			const bool lastLink = (i + 1) == m_RouteData.Links.Num();

			const SDeepDriveLane &lane = link.Lanes[0];

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

				const SDeepDriveJunction &junction = m_RoadNetwork->Junctions[link.ToJunctionId];
				const uint32 connectionSegmentId = junction.findConnectionSegment(segment.SegmentId, nextLink.Lanes[0].Segments[0]);
				if (connectionSegmentId)
					carryOverDistance = addSegmentToPoints(m_RoadNetwork->Segments[connectionSegmentId], false, carryOverDistance);
			}
		}
		UE_LOG(LogDeepDriveRoute, Log, TEXT("Route converted numPoints %d"), m_RoutePoints.Num());
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

void ADeepDriveRoute::convertToSpline(const FVector &location)
{
	if(m_RoadNetwork)
	{
		m_RouteSpline->ClearSplinePoints(true);
		m_KeySegmentMap.Empty();

		float curInputKey = 0.0f;
		if (m_RouteData.Links.Num() == 1)
		{
			const SDeepDriveRoadLink &link = m_RoadNetwork->Links[m_RouteData.Links[0]];
			curInputKey = addSegmentToSpline(m_RoadNetwork->Segments[link.Lanes[0].Segments[0]], true, curInputKey);
		}
		else
		{
			for (signed i = 0; i < m_RouteData.Links.Num(); ++i)
			{
				const SDeepDriveRoadLink &link = m_RoadNetwork->Links[m_RouteData.Links[i]];
				const bool lastLink = (i + 1) == m_RouteData.Links.Num();

				const SDeepDriveLane &lane = link.Lanes[0];

				for (signed j = 0; j < lane.Segments.Num(); ++j)
				{
					const SDeepDriveRoadSegment &segment = m_RoadNetwork->Segments[lane.Segments[j]];
					const bool lastSegment = (j + 1) == lane.Segments.Num();
					curInputKey = addSegmentToSpline(segment, lastLink && lastSegment, curInputKey);
				}

				if (!lastLink)
				{
					const SDeepDriveRoadLink &nextLink = m_RoadNetwork->Links[m_RouteData.Links[i + 1]];
					const SDeepDriveRoadSegment &segment = m_RoadNetwork->Segments[lane.Segments[lane.Segments.Num() - 1]];

					const SDeepDriveJunction &junction = m_RoadNetwork->Junctions[link.ToJunctionId];
					const uint32 connectionSegmentId = junction.findConnectionSegment(segment.SegmentId, nextLink.Lanes[0].Segments[0]);
					if (connectionSegmentId)
						curInputKey = addSegmentToSpline(m_RoadNetwork->Segments[connectionSegmentId], false, curInputKey);
				}
			}
		}

		m_RouteSpline->UpdateSpline();

		UE_LOG(LogDeepDriveRoute, Log, TEXT("Route converted key %f length %f"), curInputKey, m_RouteSpline->GetSplineLength());
	}
}

void ADeepDriveRoute::update(ADeepDriveAgent &agent)
{
	//	find closest point
	// m_BaseKey = m_RouteSpline->FindInputKeyClosestToWorldLocation(agent.GetActorLocation());

	findClosestRoutePoint(agent);

}

float ADeepDriveRoute::getRemainingDistance()
{
	return 2000.0f;
	const int32 index0 = floor(m_BaseKey);
	const int32 index1 = ceil(m_BaseKey);

	const float dist0 = m_RouteSpline->GetDistanceAlongSplineAtSplinePoint(index0);
	const float dist1 = m_RouteSpline->GetDistanceAlongSplineAtSplinePoint(index1);

	const float fac = m_BaseKey - static_cast<float>(index0);

	return FMath::Max(0.0f, m_RouteSpline->GetSplineLength() - (dist0 + fac * (dist1 - dist0)));
}

FVector ADeepDriveRoute::getLocationAhead(float distanceAhead, float sideOffset)
{
#if 0
	// skip distance points
	const float curKey = getInputKeyAhead(distanceAhead);
	FVector locAhead = m_RouteSpline->GetLocationAtSplineInputKey(curKey, ESplineCoordinateSpace::World);

	if (sideOffset != 0.0f)
	{
		FVector dir = m_RouteSpline->GetDirectionAtSplineInputKey(curKey, ESplineCoordinateSpace::World);
		dir.Z = 0.0f;
		dir.Normalize();
		FVector tng(dir.Y, -dir.X, 0.0f);
		locAhead += tng * sideOffset;
	}

#else

	int32 ind = m_curRoutePointIndex + static_cast<int32> (distanceAhead / m_StepSize);
	FVector locAhead = m_RoutePoints[ind % m_RoutePoints.Num()].Location;

#endif

	return locAhead;
}

float ADeepDriveRoute::getSpeedLimit()
{
	return 50.0f;
	const int32 key = FMath::FloorToInt(m_BaseKey);
	return m_KeySegmentMap.Contains(key) ? m_RoadNetwork->getSpeedLimit(m_KeySegmentMap[key], FMath::Frac(m_BaseKey)) : -1.0f;
}

float ADeepDriveRoute::addSegmentToSpline(const SDeepDriveRoadSegment &segment, bool addEnd, float curInputKey)
{
	if(segment.SplinePoints.Num() > 0)
	{
		const int32 count = segment.SplinePoints.Num() - 1;//(addEnd ? 0 : 1);
		for(int32 i = 1; i < count; ++i)
		{
			FSplinePoint point = segment.SplinePoints[i];
			point.InputKey = curInputKey;
			m_RouteSpline->AddPoint(point, false);

			m_KeySegmentMap.Add(static_cast<int32>(curInputKey), segment.SegmentId);
			curInputKey = curInputKey + 1.0f;
		}
	}
	else
	{
		const FVector dir = segment.EndPoint - segment.StartPoint;
		curInputKey = addSplinePoint(curInputKey, 0.15f * dir + segment.StartPoint, segment.Heading, segment.SegmentId);
		curInputKey = addSplinePoint(curInputKey, 0.5f * dir + segment.StartPoint , segment.Heading, segment.SegmentId);
		curInputKey = addSplinePoint(curInputKey, 0.85f * dir + segment.StartPoint, segment.Heading, segment.SegmentId);
		// if (addEnd)
		// 	curInputKey = addSplinePoint(curInputKey, segment.EndPoint, segment.Heading, segment.SegmentId);
	}

	return curInputKey;
}

float ADeepDriveRoute::addSplinePoint(float curInputKey, const FVector &location, float heading, float segmentId)
{
	FSplinePoint point	(	curInputKey											//	key
						,	location											//	location
						,	FVector::ZeroVector, FVector::ZeroVector			//	ArriveTng, LeaveTng
						,	FRotator(0.0f, heading, 0.0f)						//	rotator
						,	FVector::OneVector									//	scale
						,	ESplinePointType::Curve								//	type
						);

	m_RouteSpline->AddPoint(point, false);

	m_KeySegmentMap.Add(static_cast<int32>(curInputKey), segmentId);

	return curInputKey + 1.0f;
}

float ADeepDriveRoute::addSplinePoint(const SDeepDriveRoadSegment &segment, bool start, float curInputKey)
{
	FSplinePoint point	(	curInputKey											//	key
						,	start ? segment.StartPoint : segment.EndPoint		//	location
						,	FVector::ZeroVector, FVector::ZeroVector			//	ArriveTng, LeaveTng
						,	FRotator(0.0f, segment.Heading, 0.0f)				//	rotator
						,	FVector::OneVector									//	scale
						,	ESplinePointType::Curve								//	type
						);

	m_RouteSpline->AddPoint(point, false);

	m_KeySegmentMap.Add(static_cast<int32>(curInputKey), segment.SegmentId);

	return curInputKey + 1.0f;
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

float ADeepDriveRoute::getInputKeyAhead(float distanceAhead)
{
	float key = m_BaseKey;

	while (true)
	{
		const int32 index0 = floor(key);
		const int32 index1 = ceil(key);

		const float dist0 = m_RouteSpline->GetDistanceAlongSplineAtSplinePoint(index0);
		const float dist1 = m_RouteSpline->GetDistanceAlongSplineAtSplinePoint(index1);

		const float dist = (m_RouteSpline->GetLocationAtSplinePoint(index1, ESplineCoordinateSpace::World) - m_RouteSpline->GetLocationAtSplinePoint(index0, ESplineCoordinateSpace::World)).Size();

		const float relDistance = distanceAhead / dist;

		const float carryOver = key + relDistance - static_cast<float>(index1);

		if (carryOver > 0.0f)
		{
			distanceAhead -= dist * (static_cast<float>(index1) - key);
			const float newDist = (m_RouteSpline->GetLocationAtSplinePoint((index1 + 1) % m_RouteSpline->GetNumberOfSplinePoints(), ESplineCoordinateSpace::World) - m_RouteSpline->GetLocationAtSplinePoint(index1, ESplineCoordinateSpace::World)).Size();
			const float newRelDist = distanceAhead / newDist;
			key = static_cast<float>(index1) + newRelDist;
			if (newRelDist < 1.0f)
				break;
		}
		else
		{
			key += relDistance;
			break;
		}
	}

	return key;
}

FVector ADeepDriveRoute::getSplinePoint(float distance)
{
	FVector location = m_RouteSpline->GetLocationAtDistanceAlongSpline(distance, ESplineCoordinateSpace::World);
	FHitResult hitRes;
	if (GetWorld()->LineTraceSingleByChannel(hitRes, location + FVector(0.0f, 0.0f, 50.0f), location - FVector(0.0f, 0.0f, 50.0f), ECC_Visibility, FCollisionQueryParams(), FCollisionResponseParams()))
	{
		location.Z = hitRes.ImpactPoint.Z;
	}
	location.Z += 5.0f;
	return location;
}