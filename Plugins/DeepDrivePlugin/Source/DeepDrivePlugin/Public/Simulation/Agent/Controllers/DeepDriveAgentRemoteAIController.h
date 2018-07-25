

#pragma once

#include "Simulation/Agent/DeepDriveAgentControllerBase.h"
#include "DeepDriveAgentRemoteAIController.generated.h"

class ADeepDriveSplineTrack;
class ADeepDriveSimulation;

USTRUCT(BlueprintType)
struct FDeepDriveRemoteAIControllerConfiguration
{
	GENERATED_USTRUCT_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Control)
	FString		ConfigurationName;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Control)
	ADeepDriveSplineTrack	*Track;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Control)
	TArray<float>	StartDistances;

};

/**
 * 
 */
UCLASS()
class DEEPDRIVEPLUGIN_API ADeepDriveAgentRemoteAIController : public ADeepDriveAgentControllerBase
{
	GENERATED_BODY()
	
public:

	ADeepDriveAgentRemoteAIController();

	virtual void OnConfigureSimulation(const SimulationConfiguration &configuration, bool initialConfiguration);

	virtual bool Activate(ADeepDriveAgent &agent);

	virtual void SetControlValues(float steering, float throttle, float brake, bool handbrake);

	virtual bool ResetAgent();

	UFUNCTION(BlueprintCallable, Category = "Configuration")
	void Configure(const FDeepDriveRemoteAIControllerConfiguration &Configuration, int32 StartPositionSlot, ADeepDriveSimulation* DeepDriveSimulation);

private:

};
