
#pragma once

#include "Private/Utils/StateMachine/TStateMachine.hpp"
#include "Private/Utils/StateMachine/TStateBase.hpp"

class ADeepDriveAgentLocalAIController;
class ADeepDriveAgent;
class DeepDriveAgentSplineDrivingCtrl;

struct DeepDriveAgentLocalAIStateMachineContext
{
	DeepDriveAgentLocalAIStateMachineContext(ADeepDriveAgentLocalAIController &c, ADeepDriveAgent &a, DeepDriveAgentSplineDrivingCtrl &sdc)
		:	local_ai_ctrl(c)
		,	agent(a)
		,	spline_driving_ctrl(sdc)
	{	}

	ADeepDriveAgentLocalAIController		&local_ai_ctrl;
	ADeepDriveAgent							&agent;
	DeepDriveAgentSplineDrivingCtrl			&spline_driving_ctrl;

	float									side_offset = 0.0f;

};

class DeepDriveAgentLocalAIStateBase;

class DeepDriveAgentLocalAIStateMachine : public TStateMachine<DeepDriveAgentLocalAIStateMachineContext, DeepDriveAgentLocalAIStateBase>
{
public:

};


class DeepDriveAgentLocalAIStateBase : public TStateBase<DeepDriveAgentLocalAIStateMachineContext>
{
	typedef TStateBase<DeepDriveAgentLocalAIStateMachineContext> Super;

public:

	DeepDriveAgentLocalAIStateBase(DeepDriveAgentLocalAIStateMachine &stateMachine, const FString &name)
		: Super(name)
		, m_StateMachine(stateMachine)
	{
	}

	virtual ~DeepDriveAgentLocalAIStateBase()
	{
	}

protected:

	DeepDriveAgentLocalAIStateMachine		&m_StateMachine;

};
