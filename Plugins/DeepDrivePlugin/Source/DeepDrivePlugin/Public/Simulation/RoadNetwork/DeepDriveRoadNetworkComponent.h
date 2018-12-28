

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoadNetworkDefines.h"
#include "DeepDriveRoadNetworkComponent.generated.h"


class ADeepDriveJunctionProxy;
class ADeepDriveRoadLinkProxy;
class ADeepDriveAgent;

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

	//	Route CalculateRoute(const FVector Start, const FVector Destination);

	UFUNCTION(BlueprintCallable, Category = "Route")
	FVector GetRandomLocation(EDeepDriveLaneType PreferredLaneType);

	UFUNCTION(BlueprintCallable, Category = "Route")
	void PlaceAgentOnRoad(ADeepDriveAgent *Agent, bool RandomLocation);

protected:

	void collectRoadNetwork();

	SDeepDriveRoadNetwork			m_RoadNetwork;

};
