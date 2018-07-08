
#pragma once

#include "Private/Simulation/DeepDriveSimulationStateMachine.h"

class DeepDriveSimulationInitializeState : public DeepDriveSimulationStateBase
{
public:

	DeepDriveSimulationInitializeState(DeepDriveSimulationStateMachine &stateMachine);

	virtual ~DeepDriveSimulationInitializeState();

	virtual void enter(ADeepDriveSimulation &deepDriveSim);

	virtual void update(ADeepDriveSimulation &deepDriveSim, float dT);

	virtual void exit(ADeepDriveSimulation &deepDriveSim);

};
