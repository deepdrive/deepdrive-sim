

#pragma once

#include "CoreMinimal.h"
#include "Simulation/Agent/Controllers/DeepDriveAgentSplineController.h"
#include "Simulation/Agent/Controllers/LocalAI/DeepDriveAgentLocalAIStateMachine.hpp"
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

private:

	DeepDriveAgentLocalAIStateMachine			m_StateMachine;
	DeepDriveAgentLocalAIStateMachineContext	*m_StateMachineCtx;
	
};
