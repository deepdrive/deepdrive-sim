

#pragma once

#include "CoreMinimal.h"
#include "Simulation/Agent/DeepDriveAgentControllerBase.h"
#include "DeepDriveAgentCityAIController.generated.h"

class DeepDriveAgentSpeedController;
class DeepDriveAgentSteeringController;
class ADeepDriveRoute;
class ADeepDriveRoadLinkProxy;

USTRUCT(BlueprintType) struct FDeepDriveStaticRoute
{
	GENERATED_USTRUCT_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Debug)
	FVector		Start;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Debug)
	FVector		Destination;
};

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

protected:

    UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "AIProps")
	FDeepDriveCityAIControllerConfiguration		m_Configuration;

  private:

	UPROPERTY()
	ADeepDriveRoute					*m_Route = 0;

	DeepDriveAgentSpeedController		*m_SpeedController = 0;
	DeepDriveAgentSteeringController	*m_SteeringController = 0;

	int32							m_StartIndex;

	float							m_DesiredSpeed = 0.0f;

	bool							m_hasActiveGuidance = false;
};
