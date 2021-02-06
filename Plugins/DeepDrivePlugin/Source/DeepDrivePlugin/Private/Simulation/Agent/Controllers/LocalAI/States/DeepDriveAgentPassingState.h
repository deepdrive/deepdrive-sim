
#pragma once

#include "Simulation/Agent/Controllers/LocalAI/DeepDriveAgentLocalAIStateMachine.h"

class DeepDriveAgentPassingState : public DeepDriveAgentLocalAIStateBase
{
public:

	DeepDriveAgentPassingState(DeepDriveAgentLocalAIStateMachine &stateMachine);

	virtual void enter(DeepDriveAgentLocalAIStateMachineContext &ctx);

	virtual void update(DeepDriveAgentLocalAIStateMachineContext &ctx, float dT);

	virtual void exit(DeepDriveAgentLocalAIStateMachineContext &ctx);

private:

	bool abortOvertaking(DeepDriveAgentLocalAIStateMachineContext &ctx);

	ADeepDriveAgent			*m_AgentToPass = 0;

};
