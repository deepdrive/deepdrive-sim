

#pragma once

#include "CoreMinimal.h"
#include "Simulation/Agent/DeepDriveAgentControllerBase.h"
#include "DeepDriveAgentTrafficAIController.generated.h"

class DeepDriveAgentSpeedController;
class DeepDriveAgentSteeringController;
class DeepDrivePathPlanner;
class UBezierCurveComponent;
struct SDeepDrivePathConfiguration;
struct SDeepDriveRoute;

USTRUCT(BlueprintType) struct FDeepDriveTrafficAIControllerConfiguration
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
class DEEPDRIVEPLUGIN_API ADeepDriveAgentTrafficAIController : public ADeepDriveAgentControllerBase
{
	GENERATED_BODY()

public:

	ADeepDriveAgentTrafficAIController();

	virtual void Tick( float DeltaSeconds ) override;

	virtual bool Activate(ADeepDriveAgent &agent, bool keepPosition);

	virtual bool ResetAgent();

	virtual float getDistanceAlongRoute();

	virtual float getRouteLength();

	virtual float getDistanceToCenterOfTrack();

	UFUNCTION(BlueprintCallable, Category = "Configuration")
	void Configure(const FDeepDriveTrafficAIControllerConfiguration &Configuration, int32 StartPositionSlot, ADeepDriveSimulation* DeepDriveSimulation);

	UFUNCTION(BlueprintCallable, Category = "Configuration")
	void ConfigureScenario(const FDeepDriveTrafficAIControllerConfiguration &Configuration, const FDeepDriveAgentScenarioConfiguration &ScenarioConfiguration, ADeepDriveSimulation* DeepDriveSimulation);

protected:

    UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "AIProps")
	FDeepDriveTrafficAIControllerConfiguration		m_Configuration;

private:

	bool updateAgentOnPath( float DeltaSeconds);

	float checkForObstacle(float maxDistance);
	float calculateSafetyDistance();

	enum State
	{
		Idle,
		ActiveRouteGuidance,
		Waiting
	};

	UPROPERTY()
	UBezierCurveComponent				*m_BezierCurve = 0;

	DeepDriveAgentSpeedController		*m_SpeedController = 0;

	SDeepDrivePathConfiguration			*m_PathConfiguration = 0;
	DeepDrivePathPlanner				*m_PathPlanner = 0;

	int32								m_StartIndex;

	float								m_DesiredSpeed = 0.0f;

	State								m_State = Idle;

	float								m_WaitTimer = 0.0f;

	bool								m_showPath = false;

	FVector								m_maxAcceleration;
	FVector								m_totalAcceleration;
	uint32								m_numAccelerationSamples;
};
