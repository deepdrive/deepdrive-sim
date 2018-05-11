

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
		posAhead += SplineTrack->GetTangentAtSplineInputKey(curKey, ESplineCoordinateSpace::World) * sideOffset;
	}

	return posAhead;

}

float ADeepDriveSplineTrack::getSpeedLimit()
{
	return -1.0f;
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
