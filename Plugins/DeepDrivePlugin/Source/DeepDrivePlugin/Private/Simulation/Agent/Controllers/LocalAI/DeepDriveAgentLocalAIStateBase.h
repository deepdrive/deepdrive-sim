
#pragma once

#include "Private/Utils/StateMachine/TStateBase.hpp"
#include "Private/Simulation/Agent/Controllers/LocalAI/DeepDriveAgentLocalAIStateMachine.hpp"

class DeepDriveAgentLocalAIStateBase :	public TStateBase<DeepDriveAgentLocalAIStateMachineContext>
{
	typedef TStateBase<DeepDriveAgentLocalAIStateMachineContext> Super;

public:

	DeepDriveAgentLocalAIStateBase(DeepDriveAgentLocalAIStateMachine &stateMachine, const FString &name)
		:	Super(name)
		,	m_StateMachine(stateMachine)
	{
	}


protected:

	DeepDriveAgentLocalAIStateMachine		&m_StateMachine;

};
