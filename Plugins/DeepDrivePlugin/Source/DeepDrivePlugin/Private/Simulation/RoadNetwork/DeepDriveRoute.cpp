

#include "DeepDrivePluginPrivatePCH.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoute.h"
#include "Runtime/Engine/Classes/Components/SplineComponent.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"

DEFINE_LOG_CATEGORY(LogDeepDriveRoute);

// Sets default values
ADeepDriveRoute::ADeepDriveRoute()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = false;

	m_RouteSpline = CreateDefaultSubobject<USplineComponent>(TEXT("RouteSpline"));
	RootComponent = m_RouteSpline;

}

void ADeepDriveRoute::initialize(const SDeepDriveRoadNetwork &roadNetwork, const SDeepDriveRouteData &routeData)
{
	m_RoadNetwork = &roadNetwork;
	m_RouteData = routeData;
}


void ADeepDriveRoute::convert(const FVector &location)
{
	if(m_RoadNetwork)
	{
		m_RouteSpline->ClearSplinePoints(true);
		m_KeySegmentMap.Empty();

		float curInputKey = 0.0f;
		for(signed i = 0; i < m_RouteData.Links.Num(); ++i)
		{
			const SDeepDriveRoadLink &link = m_RoadNetwork->Links[m_RouteData.Links[i]];
			const bool lastLink = (i + 1) == m_RouteData.Links.Num();

			const SDeepDriveLane &lane = link.Lanes[0];

			for (signed j = 0; j < lane.Segments.Num(); ++j)
			{
				const SDeepDriveRoadSegment &segment = m_RoadNetwork->Segments[lane.Segments[j]];
				const bool lastSegment = (j + 1) == lane.Segments.Num();
				curInputKey = addSegment(segment, lastLink && lastSegment, curInputKey);
			}

			if(!lastLink)
			{
				const SDeepDriveRoadLink &nextLink = m_RoadNetwork->Links[m_RouteData.Links[i + 1]];
				const SDeepDriveRoadSegment &segment = m_RoadNetwork->Segments[lane.Segments[lane.Segments.Num() - 1]];

				const SDeepDriveJunction &junction = m_RoadNetwork->Junctions[link.ToJunctionId];
				const uint32 connectionSegmentId = junction.findConnectionSegment(segment.SegmentId, nextLink.Lanes[0].Segments[0]);
				if (connectionSegmentId)
					curInputKey = addSegment(m_RoadNetwork->Segments[connectionSegmentId], false, curInputKey);
			}
		}

		m_RouteSpline->UpdateSpline();

		UE_LOG(LogDeepDriveRoute, Log, TEXT("Route converted key %f length %f"), curInputKey, m_RouteSpline->GetSplineLength());
	}
}

void ADeepDriveRoute::update(ADeepDriveAgent &agent)
{
	m_BaseKey = m_RouteSpline->FindInputKeyClosestToWorldLocation(agent.GetActorLocation());
}

FVector ADeepDriveRoute::getLocationAhead(float distanceAhead, float sideOffset)
{
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

	return locAhead;
}

float ADeepDriveRoute::getSpeedLimit()
{
	const int32 key = FMath::FloorToInt(m_BaseKey);
	return m_KeySegmentMap.Contains(key) ? m_RoadNetwork->getSpeedLimit(m_KeySegmentMap[key], FMath::Frac(m_BaseKey)) : -1.0f;
}

float ADeepDriveRoute::addSegment(const SDeepDriveRoadSegment &segment, bool addEnd, float curInputKey)
{
	if(segment.Spline)
	{
		const int32 count = segment.Spline->Position.Points.Num() - (addEnd ? 0 : 1);
		for(int32 i = 0; i < count; ++i)
		{
			auto pos = segment.Spline->Position.Points[i];
			pos.InVal = curInputKey;
			m_RouteSpline->SplineCurves.Position.Points.Add(pos);

			auto rot = segment.Spline->Rotation.Points[i];
			rot.InVal = curInputKey;
			m_RouteSpline->SplineCurves.Rotation.Points.Add(rot);

			auto scale = segment.Spline->Scale.Points[i];
			scale.InVal = curInputKey;
			m_RouteSpline->SplineCurves.Scale.Points.Add(scale);

			m_KeySegmentMap.Add(static_cast<int32>(curInputKey), segment.SegmentId);
			curInputKey = curInputKey + 1.0f;
		}
	}
	else
	{
		curInputKey = addSplinePoint(curInputKey, segment.StartPoint, segment.Heading, segment.SegmentId);
		if(addEnd)
			curInputKey = addSplinePoint(curInputKey, segment.EndPoint, segment.Heading, segment.SegmentId);
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
