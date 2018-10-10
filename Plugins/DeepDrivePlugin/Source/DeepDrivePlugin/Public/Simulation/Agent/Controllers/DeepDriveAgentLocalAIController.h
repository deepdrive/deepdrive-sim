

#pragma once

#include "CoreMinimal.h"
#include "Components/SplineComponent.h"
#include "Simulation/Agent/DeepDriveAgentControllerBase.h"
#include "Simulation/Agent/Controllers/LocalAI/DeepDriveAgentLocalAIStateMachine.h"
#include "DeepDriveAgentLocalAIController.generated.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveAgentLocalAIController, Log, All);

class DeepDriveAgentSpeedController;
class DeepDriveAgentSteeringController;
class ADeepDriveSplineTrack;
class ADeepDriveSimulation;

USTRUCT(BlueprintType)
struct FDeepDriveLocalAIControllerConfiguration
{
	GENERATED_USTRUCT_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Control)
	FString		ConfigurationName;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Control)
	ADeepDriveSplineTrack	*Track;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Control)
	TArray<float>	StartDistances;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Control)
	FVector		PIDThrottle;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Control)
	FVector		PIDSteering;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Control)
	FVector		PIDBrake;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Control)
	FVector2D	SpeedRange;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Safety)
	float	SafetyDistanceFactor = 1.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Safety)
	float	SpeedLimitFactor = 1.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Safety)
	FVector2D	BrakingDistanceRange;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Overtaking)
	int32 MaxAgentsToOvertake = 0;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Overtaking)
	float	MinPullOutDistance = 800.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Overtaking)
	float	MinPullInDistance = 200.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Overtaking)
	float	MinSpeedDifference = 10.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Overtaking)
	float	GapBetweenAgents = 1000.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Overtaking)
	float	OvertakingOffset = 500.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Overtaking)
	float	OvertakingSpeed = 80.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Overtaking)
	float	ChangeLaneDuration = 2.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Overtaking)
	FVector4	ThinkDelays;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Overtaking)
	float	LookAheadTime = 2.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Overtaking)
	float	SafetyVsOvertakingThreshold = 0.8f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Overtaking)
	float	AbortSpeedReduction = 0.5f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Overtaking)
	float	PassingDirectionThreshold = 0.1f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Overtaking)
	float	PassingDistance = 100.0f;

};

/**
 * 
 */
UCLASS()
class DEEPDRIVEPLUGIN_API ADeepDriveAgentLocalAIController : public ADeepDriveAgentControllerBase
{
	GENERATED_BODY()
	
public:

	ADeepDriveAgentLocalAIController();

	virtual void Tick( float DeltaSeconds ) override;

	virtual bool Activate(ADeepDriveAgent &agent, bool keepPosition);

	virtual bool ResetAgent();

	virtual void OnCheckpointReached();

	virtual void OnDebugTrigger();

	UFUNCTION(BlueprintCallable, Category = "Configuration")
	void Configure(const FDeepDriveLocalAIControllerConfiguration &Configuration, int32 StartPositionSlot, ADeepDriveSimulation* DeepDriveSimulation);

	//float calculateOvertakingScore();
	//float calculateOvertakingScore(int32 maxAgentsToOvertake, float overtakingSpeed, ADeepDriveAgent* &finalAgent);
	//float calculateAbortOvertakingScore();
	
	//bool hasPassed(ADeepDriveAgent *other, float minDistance);

	float getPassedDistance(ADeepDriveAgent *other, float threshold);

	float isOppositeTrackClear(ADeepDriveAgent &nextAgent, float distanceToNextAgent, float speedDifference, float overtakingSpeed, bool considerDuration);
	float computeOppositeTrackClearance(float overtakingDistance, float speedDifference, float overtakingSpeed, bool considerDuration);

	float computeOppositeTrackClearance(float overtakingSpeed, float lookAheadDuration);

	float calculateSafetyDistance();
	//float calculateSafetyDistance(float *curDistance);

	float getDesiredSpeed() const;
	void setDesiredSpeed(float desiredSpeed);

protected:

    UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "AIProps")
	FDeepDriveLocalAIControllerConfiguration	m_Configuration;

//	UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "AIProps")
	DeepDriveAgentLocalAIStateMachineContext	*m_StateMachineCtx = 0;

//	UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "AIProps")
	DeepDriveAgentSpeedController				*m_SpeedController = 0;

//	UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "AIProps")
	DeepDriveAgentSteeringController			*m_SteeringController = 0;

//	UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "AIProps")
	ADeepDriveSplineTrack						*m_OppositeTrack = 0;

	UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "AIProps")
	float										m_DesiredSpeed;

	UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "AIProps")
	float										m_SafetyDistanceFactor = 1.0f;

	UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "AIProps")
	float										m_BrakingDeceleration = 800.0f;

    UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "AIProps")
	bool										m_isPaused = false;

private:

	DeepDriveAgentLocalAIStateMachine			m_StateMachine;

};


inline float ADeepDriveAgentLocalAIController::getDesiredSpeed() const
{
	return m_DesiredSpeed;
}

inline void ADeepDriveAgentLocalAIController::setDesiredSpeed(float desiredSpeed)
{
	m_DesiredSpeed = desiredSpeed;
}
