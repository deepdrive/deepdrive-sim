
#pragma once

#include "Private/Utils/StateMachine/TStateMachine.hpp"
#include "Private/Utils/StateMachine/TStateBase.hpp"

class ADeepDriveAgentLocalAIController;
class ADeepDriveAgent;
class DeepDriveAgentSpeedController;
class DeepDriveAgentSteeringController;

struct FDeepDriveLocalAIControllerConfiguration;

struct DeepDriveAgentLocalAIStateMachineContext
{
	DeepDriveAgentLocalAIStateMachineContext(ADeepDriveAgentLocalAIController &c, ADeepDriveAgent &a, DeepDriveAgentSpeedController &sc, DeepDriveAgentSteeringController &steering, const FDeepDriveLocalAIControllerConfiguration &cfg)
		:	local_ai_ctrl(c)
		,	agent(a)
		,	speed_controller(sc)
		,	steering_controller(steering)
		,	configuration(cfg)
	{	}

	ADeepDriveAgentLocalAIController					&local_ai_ctrl;
	ADeepDriveAgent										&agent;
	DeepDriveAgentSpeedController						&speed_controller;
	DeepDriveAgentSteeringController					&steering_controller;

	const FDeepDriveLocalAIControllerConfiguration		&configuration;

	ADeepDriveAgent										*agent_to_overtake = 0;

	float												side_offset = 0.0f;
	float												wait_time_before_overtaking = 0.0f;

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
