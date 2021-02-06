
#pragma once

#include "Simulation/Agent/Controllers/LocalAI/DeepDriveAgentLocalAIStateMachine.h"

class DeepDriveAgentPullOutState : public DeepDriveAgentLocalAIStateBase
{
public:

	DeepDriveAgentPullOutState(DeepDriveAgentLocalAIStateMachine &stateMachine);

	virtual void enter(DeepDriveAgentLocalAIStateMachineContext &ctx);

	virtual void update(DeepDriveAgentLocalAIStateMachineContext &ctx, float dT);

	virtual void exit(DeepDriveAgentLocalAIStateMachineContext &ctx);

private:

	bool abortOvertaking(DeepDriveAgentLocalAIStateMachineContext &ctx, float desiredSpeed);

	float				m_PullOutTimeFactor = 0.0f;
	float				m_PullOutAlpha = 0.0f;

	float				m_curOffset = 0.0f;

};
