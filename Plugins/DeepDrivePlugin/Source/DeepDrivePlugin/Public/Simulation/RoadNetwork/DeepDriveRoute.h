

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

	struct SRoutePoint
	{
		uint32				SegmentId;
		float				RelativePosition;

		FVector				Location;
	};

	typedef TArray<SRoutePoint>	RoutePoints;

public:	
	// Sets default values for this actor's properties
	ADeepDriveRoute();

	// Called every frame
	virtual void Tick(float DeltaTime) override;

public:	

	void initialize(const SDeepDriveRoadNetwork &roadNetwork, const SDeepDriveRouteData &routeData);

	void convert(const FVector &location);

	void update(ADeepDriveAgent &agent);

	FVector getLocationAhead(float distanceAhead, float sideOffset);

	float getSpeedLimit();

	float getRemainingDistance();

private:

	void convertToSpline(const FVector &location);
	
	void convertToPoints(const FVector &location);

	float addSegmentToSpline(const SDeepDriveRoadSegment &segment, bool addEnd, float curInputKey);
	float addSplinePoint(float curInputKey, const FVector &location, float heading, float segmentId);

	float addSegmentToPoints(const SDeepDriveRoadSegment &segment, bool addEnd, float carryOverDistance);

	float addSplinePoint(const SDeepDriveRoadSegment &segment, bool start, float curInputKey);

	float getInputKeyAhead(float distanceAhead);

	void findClosestRoutePoint(ADeepDriveAgent &agent);

	FVector getSplinePoint(float distance);

 	const SDeepDriveRoadNetwork		*m_RoadNetwork = 0;
	SDeepDriveRouteData 			m_RouteData;

	UPROPERTY()
	USplineComponent				*m_RouteSpline = 0;

	const float						m_StepSize = 100.0f;
	RoutePoints						m_RoutePoints;
	int32							m_curRoutePointIndex = -1;

	SplineKeyMap					m_KeyLinkMap;
	SplineKeyMap					m_KeySegmentMap;

	float							m_BaseKey = 0.0f;
};
