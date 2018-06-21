
#pragma once

#include "Private/Simulation/Agent/Controllers/LocalAI/DeepDriveAgentLocalAIStateMachine.h"

class DeepDriveAgentAbortOvertakingState : public DeepDriveAgentLocalAIStateBase
{
public:

	DeepDriveAgentAbortOvertakingState(DeepDriveAgentLocalAIStateMachine &stateMachine);

	virtual void enter(DeepDriveAgentLocalAIStateMachineContext &ctx);

	virtual void update(DeepDriveAgentLocalAIStateMachineContext &ctx, float dT);

	virtual void exit(DeepDriveAgentLocalAIStateMachineContext &ctx);

private:

	void fallBack(DeepDriveAgentLocalAIStateMachineContext &ctx, float dT);

	void pullIn(DeepDriveAgentLocalAIStateMachineContext &ctx, float dT);

	ADeepDriveAgent* getReferenceAgent(ADeepDriveAgent &self, float forwardDist, float backwardDist, float &distance);

	enum SubState
	{
		FallBack,
		PullIn
	};

	SubState			m_SubState;

	float				m_PullInTimeFactor = 0.0f;
	float				m_PullInAlpha = 0.0f;

	float				m_curOffset = 0.0f;

	ADeepDriveAgent		*m_OtherAgent = 0;

};
