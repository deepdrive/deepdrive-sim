
#pragma once

#include "Private/Simulation/DeepDriveSimulationStateMachine.h"

class DeepDriveSimulationResetState : public DeepDriveSimulationStateBase
{
public:

	DeepDriveSimulationResetState(DeepDriveSimulationStateMachine &stateMachine);

	virtual ~DeepDriveSimulationResetState();

	virtual void enter(ADeepDriveSimulation &deepDriveSim);

	virtual void update(ADeepDriveSimulation &deepDriveSim, float dT);

	virtual void exit(ADeepDriveSimulation &deepDriveSim);

};
