

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoadNetwork.h"
#include "DeepDriveJunctionProxy.generated.h"

class ADeepDriveRoadSegmentProxy;
class ADeepDriveRoadLinkProxy;

USTRUCT(BlueprintType)
struct FDeepDriveLaneConnectionProxy
{
	GENERATED_USTRUCT_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	ADeepDriveRoadSegmentProxy	*FromSegment = 0;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	ADeepDriveRoadSegmentProxy	*ToSegment = 0;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	ADeepDriveRoadLinkProxy *FromLink = 0;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	ADeepDriveRoadLinkProxy *ToLink = 0;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	ADeepDriveRoadSegmentProxy	*ConnectionSegment = 0;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	float	SpeedLimit = 15.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	float	SlowDownDistance = 1000.0f;

};


UCLASS()
class DEEPDRIVEPLUGIN_API ADeepDriveJunctionProxy : public AActor
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	ADeepDriveJunctionProxy();

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

	virtual bool ShouldTickIfViewportsOnly() const override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

	const TArray<ADeepDriveRoadLinkProxy*>& getLinksIn();

	const TArray<ADeepDriveRoadLinkProxy*>& getLinksOut();

	const TArray<FDeepDriveLaneConnectionProxy>& getLaneConnections();

protected:

	UPROPERTY(EditDefaultsOnly, Category = Default)
	USceneComponent				*Root = 0;

	UPROPERTY(EditAnywhere, Category = Configuration)
	TArray<ADeepDriveRoadLinkProxy*>	LinksIn;

	UPROPERTY(EditAnywhere, Category = Configuration)
	TArray<ADeepDriveRoadLinkProxy*>	LinksOut;

	UPROPERTY(EditAnywhere, Category = Configuration)
	TArray<FDeepDriveLaneConnectionProxy>	LaneConnections;

	UPROPERTY(EditAnywhere, Category = Debug)
	FColor						Color = FColor(0, 255, 0, 128);

	bool						m_IsGameRunning = false;

};

inline const TArray<ADeepDriveRoadLinkProxy*>& ADeepDriveJunctionProxy::getLinksIn()
{
	return LinksIn;
}

inline const TArray<ADeepDriveRoadLinkProxy*>& ADeepDriveJunctionProxy::getLinksOut()
{
	return LinksOut;
}

inline const TArray<FDeepDriveLaneConnectionProxy>& ADeepDriveJunctionProxy::getLaneConnections()
{
	return LaneConnections;
}
