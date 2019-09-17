

#pragma once

#include "CoreMinimal.h"
#include "Simulation/Agent/DeepDriveAgentControllerBase.h"
#include "DeepDriveAgentCityAIController.generated.h"

class DeepDriveAgentSpeedController;
class DeepDriveAgentSteeringController;
class ADeepDriveRoadLinkProxy;

USTRUCT(BlueprintType) struct FDeepDriveCityAIControllerConfiguration
{
	GENERATED_USTRUCT_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Control)
	FString		ConfigurationName;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Debug)
	TArray<FDeepDriveStaticRoute> Routes;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Control)
	TArray<FVector>		StartPositions;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Control)
	FVector PIDThrottle;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Control)
	FVector PIDSteering;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Control)
	FVector PIDBrake;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Control)
	FVector2D SpeedRange;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Safety)
	float SafetyDistanceFactor = 1.0f;

};

/**
 * 
 */
UCLASS()
class DEEPDRIVEPLUGIN_API ADeepDriveAgentCityAIController : public ADeepDriveAgentControllerBase
{
	GENERATED_BODY()

public:

	ADeepDriveAgentCityAIController();

	virtual void Tick( float DeltaSeconds ) override;

	virtual bool Activate(ADeepDriveAgent &agent, bool keepPosition);

	virtual bool ResetAgent();

	UFUNCTION(BlueprintCallable, Category = "Configuration")
	void Configure(const FDeepDriveCityAIControllerConfiguration &Configuration, int32 StartPositionSlot, ADeepDriveSimulation* DeepDriveSimulation);

	UFUNCTION(BlueprintCallable, Category = "Configuration")
	void ConfigureScenario(const FDeepDriveCityAIControllerConfiguration &Configuration, const FDeepDriveAgentScenarioConfiguration &ScenarioConfiguration, ADeepDriveSimulation* DeepDriveSimulation);

protected:

    UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "AIProps")
	FDeepDriveCityAIControllerConfiguration		m_Configuration;

private:

	float checkForObstacle(float maxDistance);
	float calculateSafetyDistance();

	enum State
	{
		Idle,
		ActiveRouteGuidance,
		Waiting
	};

	DeepDriveAgentSpeedController		*m_SpeedController = 0;
	DeepDriveAgentSteeringController	*m_SteeringController = 0;

	int32								m_StartIndex;

	float								m_DesiredSpeed = 0.0f;

	State								m_State = Idle;

	float								m_WaitTimer = 0.0f;
};
