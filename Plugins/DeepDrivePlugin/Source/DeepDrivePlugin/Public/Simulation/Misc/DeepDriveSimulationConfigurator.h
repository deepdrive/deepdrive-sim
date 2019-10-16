

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "DeepDriveSimulationConfigurator.generated.h"

class ADeepDriveRoadLinkProxy;
class ADeepDriveRoadSegmentProxy;
class ADeepDriveAgent;
class ADeepDriveSimulation;

USTRUCT(BlueprintType)
struct FDeepDriveRoadLocation
{
	GENERATED_USTRUCT_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	ADeepDriveRoadLinkProxy	*Link = 0;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	ADeepDriveRoadSegmentProxy	*Segment = 0;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	float	RelativePosition = 0.5f;
};

USTRUCT(BlueprintType)
struct FDeepDriveSimulationScenarioAgent
{
	GENERATED_USTRUCT_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	FDeepDriveRoadLocation	Start;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	FDeepDriveRoadLocation	Destination;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	float	MinSpeed = 0.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	float	MaxSpeed = 0.0f;
};

USTRUCT(BlueprintType)
struct FDeepDriveSimulationScenario
{
	GENERATED_USTRUCT_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	FString		Title;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	FDeepDriveSimulationScenarioAgent	EgoAgent;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	TArray<FDeepDriveSimulationScenarioAgent>	AdditionalAgents;
};

UCLASS()
class DEEPDRIVEPLUGIN_API ADeepDriveSimulationConfigurator : public AActor
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	ADeepDriveSimulationConfigurator();

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Simulation)
	ADeepDriveSimulation *Simulation = 0;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Configuration)
	TArray< TSubclassOf<ADeepDriveAgent> >		Agents;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Configuration)
	TArray<FDeepDriveSimulationScenario>	Scenarios;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Simulation)
	int32	ScenarioIndex = 0;

private:

	FVector resolvePosition(const FDeepDriveRoadLocation &roadLocation);

	int32				m_StartCounter = 0;
};
