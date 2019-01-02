

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoadNetwork.h"
#include "DeepDriveRoute.generated.h"

class ADeepDriveAgent;

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveRoute, Log, All);

UCLASS()
class DEEPDRIVEPLUGIN_API ADeepDriveRoute : public AActor
{
	GENERATED_BODY()

private:

	typedef TMap<int32, uint32>		SplineKeyMap;

public:	
	// Sets default values for this actor's properties
	ADeepDriveRoute();

public:	

	void initialize(const SDeepDriveRoadNetwork &roadNetwork, const SDeepDriveRouteData &routeData);

	void convert(const FVector &location);

	void update(ADeepDriveAgent &agent);

	FVector getLocationAhead(float distanceAhead, float sideOffset);

	float getSpeedLimit();

private:

	float addSegment(const SDeepDriveRoadSegment &segment, bool addEnd, float curInputKey);
	float addSplinePoint(float curInputKey, const FVector &location, float heading, float segmentId);

	float addSplinePoint(const SDeepDriveRoadSegment &segment, bool start, float curInputKey);

	float getInputKeyAhead(float distanceAhead);

 	const SDeepDriveRoadNetwork		*m_RoadNetwork = 0;
	SDeepDriveRouteData 			m_RouteData;

	UPROPERTY()
	USplineComponent				*m_RouteSpline = 0;

	SplineKeyMap					m_KeyLinkMap;
	SplineKeyMap					m_KeySegmentMap;

	float							m_BaseKey = 0.0f;
};
