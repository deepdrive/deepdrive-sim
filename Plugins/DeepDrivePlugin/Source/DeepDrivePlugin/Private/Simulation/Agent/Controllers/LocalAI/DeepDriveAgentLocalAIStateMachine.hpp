
#pragma once

#include "Private/Utils/StateMachine/TStateMachine.hpp"

class ADeepDriveAgentLocalAIController;

struct DeepDriveAgentLocalAIStateMachineContext
{
	DeepDriveAgentLocalAIStateMachineContext(ADeepDriveAgentLocalAIController &c, ADeepDriveAgent &a)
		:	local_ai_ctrl(c)
		,	agent(a)
	{	}

	ADeepDriveAgentLocalAIController		&local_ai_ctrl;
	ADeepDriveAgent							&agent;
};

class DeepDriveAgentLocalAIStateBase;

class DeepDriveAgentLocalAIStateMachine : public TStateMachine<DeepDriveAgentLocalAIStateMachineContext, DeepDriveAgentLocalAIStateBase>
{
public:

};