

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Components/SplineComponent.h"
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

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

	FVector getStartPoint();
	FVector getEndPoint();

	const FSplineCurves* getSplineCurves();

	const TArray<FVector2D>& getSpeedLimits();

protected:

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
	TArray<FVector2D> SpeedLimits;

};


inline FVector ADeepDriveRoadSegmentProxy::getStartPoint()
{
	return StartPoint->GetComponentLocation();
}

inline FVector ADeepDriveRoadSegmentProxy::getEndPoint()
{
	return EndPoint->GetComponentLocation();
}

inline const FSplineCurves* ADeepDriveRoadSegmentProxy::getSplineCurves()
{
	return Spline ? &Spline->SplineCurves : 0;
}

inline const TArray<FVector2D>& ADeepDriveRoadSegmentProxy::getSpeedLimits()
{
	return SpeedLimits;
}
