

#pragma once

#include "CoreMinimal.h"
#include "Simulation/Agent/Controllers/DeepDriveAgentSplineController.h"
#include "Simulation/Agent/Controllers/LocalAI/DeepDriveAgentLocalAIStateMachine.h"
#include "DeepDriveAgentLocalAIController.generated.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveAgentLocalAIController, Log, All);



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
	FVector2D	BrakingDistanceRange;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Overtaking)
	bool OvertakingEnabled = false;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Overtaking)
	float	OvertakingMinDistance = 800.0f;

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

};

/**
 * 
 */
UCLASS()
class DEEPDRIVEPLUGIN_API ADeepDriveAgentLocalAIController : public ADeepDriveAgentSplineController
{
	GENERATED_BODY()
	
public:

	ADeepDriveAgentLocalAIController();

	virtual void Tick( float DeltaSeconds ) override;

	virtual bool Activate(ADeepDriveAgent &agent);

	UFUNCTION(BlueprintCallable, Category = "Configuration")
	void Configure(const FDeepDriveLocalAIControllerConfiguration &Configuration, int32 StartPositionSlot);

private:

	void think(float dT);

	float calculateOvertakingScore(ADeepDriveAgent &nextAgent, float distanceToNextAgent);

	DeepDriveAgentLocalAIStateMachine			m_StateMachine;
	DeepDriveAgentLocalAIStateMachineContext	*m_StateMachineCtx;
	
	FDeepDriveLocalAIControllerConfiguration	m_Configuration;

};
