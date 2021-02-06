
#pragma once

#include "Simulation/DeepDriveSimulationStateMachine.h"

class DeepDriveSimulationInitializeState : public DeepDriveSimulationStateBase
{
public:

	DeepDriveSimulationInitializeState(DeepDriveSimulationStateMachine &stateMachine, bool scenarioMode);

	virtual ~DeepDriveSimulationInitializeState();

	virtual void enter(ADeepDriveSimulation &deepDriveSim);

	virtual void update(ADeepDriveSimulation &deepDriveSim, float dT);

	virtual void exit(ADeepDriveSimulation &deepDriveSim);

private:

	bool			m_ScenarioMode;
	
};
