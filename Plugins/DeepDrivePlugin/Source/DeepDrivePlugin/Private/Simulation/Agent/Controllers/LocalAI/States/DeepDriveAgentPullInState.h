
#pragma once

#include "Private/Simulation/Agent/Controllers/LocalAI/DeepDriveAgentLocalAIStateMachine.h"

class DeepDriveAgentPullInState : public DeepDriveAgentLocalAIStateBase
{
public:

	DeepDriveAgentPullInState(DeepDriveAgentLocalAIStateMachine &stateMachine);

	virtual void enter(DeepDriveAgentLocalAIStateMachineContext &ctx);

	virtual void update(DeepDriveAgentLocalAIStateMachineContext &ctx, float dT);

	virtual void exit(DeepDriveAgentLocalAIStateMachineContext &ctx);

private:

	float				m_remainingPullInTime = 0.0f;

	float				m_curOffset = 0.0f;
	float				m_deltaOffsetFac = 0.0f;

};
