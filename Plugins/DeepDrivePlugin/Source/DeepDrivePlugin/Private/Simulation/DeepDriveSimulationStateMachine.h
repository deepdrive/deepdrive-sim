
#pragma once

#include "Utils/StateMachine/TStateMachine.hpp"
#include "Utils/StateMachine/TStateBase.hpp"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveSimulationState, Log, All);

class ADeepDriveSimulation;

class DeepDriveSimulationStateBase;

struct DeepDriveSimulationStateMachineContext
{

};

class DeepDriveSimulationStateMachine : public TStateMachine<ADeepDriveSimulation, DeepDriveSimulationStateBase>
{
public:

};


class DeepDriveSimulationStateBase : public TStateBase<ADeepDriveSimulation>
{
	typedef TStateBase<ADeepDriveSimulation> Super;

public:

	DeepDriveSimulationStateBase(DeepDriveSimulationStateMachine &stateMachine, const FString &name)
		: Super(name)
		, m_StateMachine(stateMachine)
	{
	}

	virtual ~DeepDriveSimulationStateBase()
	{
	}

protected:

	DeepDriveSimulationStateMachine			&m_StateMachine;

};
