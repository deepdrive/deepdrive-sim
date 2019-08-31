

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoadNetwork.h"
#include "DeepDriveRoadNetworkComponent.generated.h"

class ADeepDriveSimulation;
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

	void Initialize(ADeepDriveSimulation &deepDriveSim);

	UFUNCTION(BlueprintCallable, Category = "Route")
	FVector GetRandomLocation(EDeepDriveLaneType PreferredLaneType, int32 relPos);

	UFUNCTION(BlueprintCallable, Category = "Route")
	void PlaceAgentOnRoad(ADeepDriveAgent *Agent, const FVector &Location, bool OnClosestSegment);

	UFUNCTION(BlueprintCallable, Category = "Route")
	void PlaceAgentOnRoadRandomly(ADeepDriveAgent *Agent);

	TArray<uint32> CalculateRoute(const FVector Start, const FVector Destination);

	ADeepDriveRoute *CalculateRoute(const TArray<uint32> &routeLinks);

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Debug)
	bool	ShowRoutes = false;

	uint32 getRoadLink(ADeepDriveRoadLinkProxy *linkProxy);

	TArray<uint32> calculateRandomRoute(const FVector &Start, FVector &Destination);

	const SDeepDriveRoadNetwork& getRoadNetwork();

	uint32 getRandomRoadLink(bool validFromJunction, bool validToJunction);

protected:


	void collectRoadNetwork();

	float calcHeading(const FVector &from, const FVector &to);

	ADeepDriveSimulation			*m_DeepDriveSimulation = 0;
	SDeepDriveRoadNetwork			m_RoadNetwork;

	DeepDriveRoadNetworkExtractor	*m_Extractor;

};


inline const SDeepDriveRoadNetwork& UDeepDriveRoadNetworkComponent::getRoadNetwork()
{
	return m_RoadNetwork;
}
