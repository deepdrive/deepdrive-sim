
#pragma once

#include "Private/Simulation/Agent/Controllers/LocalAI/DeepDriveAgentLocalAIStateMachine.h"

class DeepDriveAgentPassingState : public DeepDriveAgentLocalAIStateBase
{
public:

	DeepDriveAgentPassingState(DeepDriveAgentLocalAIStateMachine &stateMachine);

	virtual void enter(DeepDriveAgentLocalAIStateMachineContext &ctx);

	virtual void update(DeepDriveAgentLocalAIStateMachineContext &ctx, float dT);

	virtual void exit(DeepDriveAgentLocalAIStateMachineContext &ctx);

private:

	bool abortOvertaking(DeepDriveAgentLocalAIStateMachineContext &ctx);

	ADeepDriveAgent				*m_AgentToPass = 0;

	float						m_totalSpeedDifference = 0.0f;
	uint32						m_totalSpeedDifferenceCount = 0;

	float						m_TotalOppositeTrackClearance = 0.0f;
	float						m_OppositeTrackClearanceCount = 0.0f;

};
