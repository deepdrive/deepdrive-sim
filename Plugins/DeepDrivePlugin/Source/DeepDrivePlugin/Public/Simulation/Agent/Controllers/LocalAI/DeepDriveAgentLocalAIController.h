

#pragma once

#include "CoreMinimal.h"
#include "Simulation/Agent/Controllers/DeepDriveAgentSplineController.h"
#include "Simulation/Agent/Controllers/LocalAI/DeepDriveAgentLocalAIStateMachine.h"
#include "DeepDriveAgentLocalAIController.generated.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveAgentLocalAIController, Log, All);

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


	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Overtaking)
	float	OvertakingOffset = 500.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Overtaking)
	float	OvertakingSpeed = 80.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Overtaking)
	float	OvertakingBeginDuration = 4.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Overtaking)
	float	ChangeLaneDuration = 2.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Overtaking)
	float	OvertakingDuration = 10.0f;

private:

	DeepDriveAgentLocalAIStateMachine			m_StateMachine;
	DeepDriveAgentLocalAIStateMachineContext	*m_StateMachineCtx;
	
};
