

#include "DeepDrivePluginPrivatePCH.h"
#include "DeepDriveSplineTrack.h"
#include "Runtime/Engine/Classes/Components/SplineComponent.h"

ADeepDriveSplineTrack::ADeepDriveSplineTrack()
	:	SplineTrack(0)
{
	SplineTrack = CreateDefaultSubobject<USplineComponent>(TEXT("SplneTrack"));
	RootComponent = SplineTrack;
//	SplineTrack->SetupAttachment(GetRootComponent());
}

ADeepDriveSplineTrack::~ADeepDriveSplineTrack()
{
}

void ADeepDriveSplineTrack::BeginPlay()
{
	for (auto &x : SpeedLimits)
	{
		const float dist = x.X;
		const float key = SplineTrack->GetInputKeyAtDistanceAlongSpline(dist);

		const FVector loc = SplineTrack->GetLocationAtDistanceAlongSpline(dist, ESplineCoordinateSpace::World);

		m_SpeedLimits.Add(FVector2D(SplineTrack->FindInputKeyClosestToWorldLocation(loc), x.Y));
	}

	m_SpeedLimits.Sort([](const FVector2D &lhs, const FVector2D &rhs) { return lhs.X < rhs.X; });
}

void ADeepDriveSplineTrack::setBaseLocation(const FVector &baseLocation)
{
	m_BaseLocation = baseLocation;
	m_BaseKey = SplineTrack->FindInputKeyClosestToWorldLocation(baseLocation);
}


FVector ADeepDriveSplineTrack::getLocationAhead(float distanceAhead, float sideOffset)
{
	const float curKey = getInputKeyAhead(distanceAhead);
	FVector posAhead = SplineTrack->GetLocationAtSplineInputKey(curKey, ESplineCoordinateSpace::World);

	if (sideOffset != 0.0f)
	{
		FVector dir = SplineTrack->GetDirectionAtSplineInputKey(curKey, ESplineCoordinateSpace::World);
		dir.Z = 0.0f;
		dir.Normalize();
		FVector tng(dir.Y, -dir.X, 0.0f);
		posAhead += tng * sideOffset;
	}

	return posAhead;

}

float ADeepDriveSplineTrack::getSpeedLimit(float distanceAhead)
{
	float speedLimit = -1.0f;

	if (m_SpeedLimits.Num() > 0)
	{
		const float curKey = distanceAhead > 0.0f ? getInputKeyAhead(distanceAhead) : m_BaseKey;
		if	(	curKey < m_SpeedLimits[0].X
			||	curKey >= m_SpeedLimits.Last().X
			)
		{
			speedLimit = m_SpeedLimits.Last().Y;
		}
		else
		{
			for (signed i = 0; i < m_SpeedLimits.Num(); ++i)
			{
				if (curKey >= m_SpeedLimits[i].X)
				{
					speedLimit = m_SpeedLimits[i].Y;
				}
				else
					break;
			}
		}
	}

	return speedLimit;
}

float ADeepDriveSplineTrack::getInputKeyAhead(float distanceAhead)
{
	float key = m_BaseKey;

	while (true)
	{
		const int32 index0 = floor(key);
		const int32 index1 = ceil(key);

		const float dist0 = SplineTrack->GetDistanceAlongSplineAtSplinePoint(index0);
		const float dist1 = SplineTrack->GetDistanceAlongSplineAtSplinePoint(index1);

		const float dist = (SplineTrack->GetLocationAtSplinePoint(index1, ESplineCoordinateSpace::World) - SplineTrack->GetLocationAtSplinePoint(index0, ESplineCoordinateSpace::World)).Size();

		const float relDistance = distanceAhead / dist;

		const float carryOver = key + relDistance - static_cast<float> (index1);

		if (carryOver > 0.0f)
		{
			distanceAhead -= dist * (static_cast<float> (index1) - key);
			const float newDist = (SplineTrack->GetLocationAtSplinePoint((index1 + 1) % SplineTrack->GetNumberOfSplinePoints(), ESplineCoordinateSpace::World) - SplineTrack->GetLocationAtSplinePoint(index1, ESplineCoordinateSpace::World)).Size();
			const float newRelDist = distanceAhead / newDist;
			key = static_cast<float> (index1) + newRelDist;
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
