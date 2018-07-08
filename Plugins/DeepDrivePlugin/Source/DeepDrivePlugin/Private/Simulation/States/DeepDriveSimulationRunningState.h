
#pragma once

#include "Private/Simulation/DeepDriveSimulationStateMachine.h"

class DeepDriveSimulationRunningState : public DeepDriveSimulationStateBase
{
public:

	DeepDriveSimulationRunningState(DeepDriveSimulationStateMachine &stateMachine);

	virtual ~DeepDriveSimulationRunningState();

	virtual void enter(ADeepDriveSimulation &deepDriveSim);

	virtual void update(ADeepDriveSimulation &deepDriveSim, float dT);

	virtual void exit(ADeepDriveSimulation &deepDriveSim);

};
