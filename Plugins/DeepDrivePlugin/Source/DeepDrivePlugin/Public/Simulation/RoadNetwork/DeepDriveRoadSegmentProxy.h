

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Components/SplineComponent.h"
#include "Simulation/RoadNetwork/DeepDriveRoadNetwork.h"
#include "DeepDriveRoadSegmentProxy.generated.h"

UCLASS()
class DEEPDRIVEPLUGIN_API ADeepDriveRoadSegmentProxy : public AActor
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	ADeepDriveRoadSegmentProxy();

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

	virtual bool ShouldTickIfViewportsOnly() const override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

	FVector getStartPoint();
	FVector getEndPoint();

	const USplineComponent* getSpline();
	const FSplineCurves* getSplineCurves();

	float getSpeedLimit();

	EDeepDriveConnectionShape getConnectionShape();

protected:

	FVector getSplinePoint(int32 index);

	UPROPERTY(EditDefaultsOnly, Category = Default)
	USceneComponent				*Root = 0;

	UPROPERTY(EditDefaultsOnly, Category = Default)
	UArrowComponent				*StartPoint = 0;

	UPROPERTY(EditDefaultsOnly, Category = Default)
	UArrowComponent				*EndPoint = 0;

	UPROPERTY(EditDefaultsOnly, Category = Default)
	USplineComponent			*Spline = 0;
	
	UPROPERTY(EditAnywhere, Category = Configuration)
	ADeepDriveRoadSegmentProxy	*LeftLane = 0;

	UPROPERTY(EditAnywhere, Category = Configuration)
	ADeepDriveRoadSegmentProxy	*RightLane = 0;

	UPROPERTY(EditAnywhere, Category = Configuration)
	float	SpeedLimit = -1.0f;

	UPROPERTY(EditAnywhere, Category = Debug)
	FColor						Color = FColor::Red;

	bool						m_IsGameRunning = false;

	FVector						m_LastStartLocation = FVector::ZeroVector;
	FRotator					m_LastStartRotation = FRotator(0.0f, 0.0f, 0.0f);

	FVector						m_LastEndLocation = FVector::ZeroVector;
	FRotator					m_LastEndRotatior = FRotator(0.0f, 0.0f, 0.0f);
};


inline FVector ADeepDriveRoadSegmentProxy::getStartPoint()
{
	return StartPoint->GetComponentLocation();
}

inline FVector ADeepDriveRoadSegmentProxy::getEndPoint()
{
	return EndPoint->GetComponentLocation();
}

inline const USplineComponent* ADeepDriveRoadSegmentProxy::getSpline()
{
	return Spline;
}

inline const FSplineCurves* ADeepDriveRoadSegmentProxy::getSplineCurves()
{
	return Spline ? &Spline->SplineCurves : 0;
}

inline float ADeepDriveRoadSegmentProxy::getSpeedLimit()
{
	return SpeedLimit;
}
