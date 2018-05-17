
#pragma once

#include "Private/Simulation/Agent/Controllers/LocalAI/DeepDriveAgentLocalAIStateMachine.h"

class DeepDriveAgentOvertakingState : public DeepDriveAgentLocalAIStateBase
{
public:

	DeepDriveAgentOvertakingState(DeepDriveAgentLocalAIStateMachine &stateMachine);

	virtual void enter(DeepDriveAgentLocalAIStateMachineContext &ctx);

	virtual void update(DeepDriveAgentLocalAIStateMachineContext &ctx, float dT);

	virtual void exit(DeepDriveAgentLocalAIStateMachineContext &ctx);

private:

	float				m_remainingOvertakingTime = 0.0f;
};
