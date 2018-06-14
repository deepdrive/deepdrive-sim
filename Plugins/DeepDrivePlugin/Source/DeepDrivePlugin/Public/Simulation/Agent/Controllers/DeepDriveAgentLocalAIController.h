

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

	virtual bool Activate(ADeepDriveAgent &agent);

	virtual bool ResetAgent();

	virtual void OnCheckpointReached();

	virtual void OnDebugTrigger();

	UFUNCTION(BlueprintCallable, Category = "Configuration")
	void Configure(const FDeepDriveLocalAIControllerConfiguration &Configuration, int32 StartPositionSlot);

	//float calculateOvertakingScore();
	//float calculateOvertakingScore(int32 maxAgentsToOvertake, float overtakingSpeed, ADeepDriveAgent* &finalAgent);
	//float calculateAbortOvertakingScore();
	
	//bool hasPassed(ADeepDriveAgent *other, float minDistance);

	float getPassedDistance(ADeepDriveAgent *other);

	float isOppositeTrackClear(ADeepDriveAgent &nextAgent, float distanceToNextAgent, float speedDifference, float overtakingSpeed, bool considerDuration);
	float computeOppositeTrackClearance(float overtakingDistance, float speedDifference, float overtakingSpeed, bool considerDuration);

	float calculateSafetyDistance();
	//float calculateSafetyDistance(float *curDistance);

	float getDesiredSpeed() const;
	void setDesiredSpeed(float desiredSpeed);

private:

	void resetAgentPosOnSpline(ADeepDriveAgent &agent);
	float getClosestDistanceOnSpline(const FVector &location);

	DeepDriveAgentLocalAIStateMachine			m_StateMachine;
	DeepDriveAgentLocalAIStateMachineContext	*m_StateMachineCtx = 0;
	
	DeepDriveAgentSpeedController				*m_SpeedController = 0;
	DeepDriveAgentSteeringController			*m_SteeringController = 0;

	FDeepDriveLocalAIControllerConfiguration	m_Configuration;

	ADeepDriveSplineTrack						*m_Track = 0;
	ADeepDriveSplineTrack						*m_OppositeTrack = 0;
	float										m_StartDistance = 0.0f;
	float										m_DesiredSpeed;

	USplineComponent							*m_Spline = 0;
	float										m_SafetyDistanceFactor = 1.0f;
	float										m_BrakingDeceleration = 800.0f;

	bool										m_isPaused = false;

};


inline float ADeepDriveAgentLocalAIController::getDesiredSpeed() const
{
	return m_DesiredSpeed;
}

inline void ADeepDriveAgentLocalAIController::setDesiredSpeed(float desiredSpeed)
{
	m_DesiredSpeed = desiredSpeed;
}
