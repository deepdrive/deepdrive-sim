
#pragma once

#include "Private/Simulation/Agent/Controllers/LocalAI/DeepDriveAgentLocalAIStateMachine.h"

class DeepDriveAgentCruiseState : public DeepDriveAgentLocalAIStateBase
{
public:

	DeepDriveAgentCruiseState(DeepDriveAgentLocalAIStateMachine &stateMachine);

	virtual void enter(DeepDriveAgentLocalAIStateMachineContext &ctx);

	virtual void update(DeepDriveAgentLocalAIStateMachineContext &ctx, float dT);

	virtual void exit(DeepDriveAgentLocalAIStateMachineContext &ctx);

private:

	bool isOvertakingPossible(DeepDriveAgentLocalAIStateMachineContext &ctx);

	float				m_WaitTimeBeforeOvertaking;
};
