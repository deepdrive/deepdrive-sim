

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoadNetwork.h"
#include "DeepDriveRoadLinkProxy.generated.h"

class ADeepDriveRoadSegmentProxy;

USTRUCT(BlueprintType)
struct FDeepDriveLaneProxy
{
	GENERATED_USTRUCT_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	EDeepDriveLaneType	LaneType;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	TArray<ADeepDriveRoadSegmentProxy*>		Segments;

};

UCLASS()
class DEEPDRIVEPLUGIN_API ADeepDriveRoadLinkProxy : public AActor
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	ADeepDriveRoadLinkProxy();

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

	FVector getStartPoint();
	FVector getEndPoint();
	
	const TArray<FDeepDriveLaneProxy>& getLanes();

	float getSpeedLimit();

protected:

	UPROPERTY(EditDefaultsOnly, Category = Default)
	USceneComponent				*Root = 0;

	UPROPERTY(EditDefaultsOnly, Category = Default)
	UArrowComponent				*StartPoint = 0;

	UPROPERTY(EditDefaultsOnly, Category = Default)
	UArrowComponent				*EndPoint = 0;

	UPROPERTY(EditAnywhere, Category = Configuration)
	TArray<FDeepDriveLaneProxy>	Lanes;

	UPROPERTY(EditAnywhere, Category = Configuration)
	float	SpeedLimit = DeepDriveRoadNetwork::SpeedLimitInTown;

};

inline FVector ADeepDriveRoadLinkProxy::getStartPoint()
{
	return StartPoint->GetComponentLocation();
}

inline FVector ADeepDriveRoadLinkProxy::getEndPoint()
{
	return EndPoint->GetComponentLocation();
}

inline const TArray<FDeepDriveLaneProxy>& ADeepDriveRoadLinkProxy::getLanes()
{
	return Lanes;
}

inline float ADeepDriveRoadLinkProxy::getSpeedLimit()
{
	return SpeedLimit;
}
