
#pragma once

#include "Private/Utils/StateMachine/TStateMachine.hpp"
#include "Private/Utils/StateMachine/TStateBase.hpp"

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
