

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoadNetwork.h"
#include "DeepDriveRoadNetworkComponent.generated.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveRoadNetwork, Log, All);

class ADeepDriveJunctionProxy;
class ADeepDriveRoadLinkProxy;
class ADeepDriveAgent;
class ADeepDriveRoute;
class DeepDriveRoadNetworkExtractor;

UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent) )
class DEEPDRIVEPLUGIN_API UDeepDriveRoadNetworkComponent : public UActorComponent
{
	GENERATED_BODY()

public:	
	// Sets default values for this component's properties
	UDeepDriveRoadNetworkComponent();

protected:
	// Called when the game starts
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

	void Initialize();

	UFUNCTION(BlueprintCallable, Category = "Route")
	FVector GetRandomLocation(EDeepDriveLaneType PreferredLaneType);

	UFUNCTION(BlueprintCallable, Category = "Route")
	void PlaceAgentOnRoad(ADeepDriveAgent *Agent, FVector Location);

	UFUNCTION(BlueprintCallable, Category = "Route")
	void PlaceAgentOnRoadRandomly(ADeepDriveAgent *Agent);

	UFUNCTION(BlueprintCallable, Category = "Route")
	ADeepDriveRoute* CalculateRoute(const FVector Start, const FVector Destination);

	ADeepDriveRoute *CalculateRoute(const TArray<uint32> &routeLinks);

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Debug)
	TArray<ADeepDriveRoadLinkProxy*>	Route;

	uint32 getRoadLink(ADeepDriveRoadLinkProxy *linkProxy);

protected:

	void collectRoadNetwork();

	SDeepDriveRoadLink* findClosestLink(const FVector &pos);

	float calcHeading(const FVector &from, const FVector &to);

	SDeepDriveRoadNetwork		m_RoadNetwork;

	DeepDriveRoadNetworkExtractor	*m_Extractor;

	TArray<uint32>				m_DebugRoute;

};
