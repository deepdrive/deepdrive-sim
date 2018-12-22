

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "DeepDriveRoadSegmentProxy.generated.h"

class USplineComponent;

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

};
