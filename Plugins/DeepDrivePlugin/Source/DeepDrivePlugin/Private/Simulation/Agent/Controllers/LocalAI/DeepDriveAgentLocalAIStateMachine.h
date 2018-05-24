
#pragma once

#include "Private/Utils/StateMachine/TStateMachine.hpp"
#include "Private/Utils/StateMachine/TStateBase.hpp"

class ADeepDriveAgentLocalAIController;
class ADeepDriveAgent;
class DeepDriveAgentSplineDrivingCtrl;

struct FDeepDriveLocalAIControllerConfiguration;

struct DeepDriveAgentLocalAIStateMachineContext
{
	DeepDriveAgentLocalAIStateMachineContext(ADeepDriveAgentLocalAIController &c, ADeepDriveAgent &a, DeepDriveAgentSplineDrivingCtrl &sdc, const FDeepDriveLocalAIControllerConfiguration &cfg)
		:	local_ai_ctrl(c)
		,	agent(a)
		,	spline_driving_ctrl(sdc)
		,	configuration(cfg)
	{	}

	ADeepDriveAgentLocalAIController					&local_ai_ctrl;
	ADeepDriveAgent										&agent;
	DeepDriveAgentSplineDrivingCtrl						&spline_driving_ctrl;
	const FDeepDriveLocalAIControllerConfiguration		&configuration;

	ADeepDriveAgent										*next_agent = 0;
	float												distance_to_next_agent = -1.0f;

	float												overtaking_score = -1.0f;
	bool												overtaking_in_progess = false;

	float												side_offset = 0.0f;

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
