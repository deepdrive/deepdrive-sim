
#include "Simulation/States/DeepDriveSimulationRunningState.h"
#include "Simulation/DeepDriveSimulation.h"
#include "Simulation/DeepDriveSimulationServerProxy.h"
#include "Simulation/DeepDriveSimulationCaptureProxy.h"

DeepDriveSimulationRunningState::DeepDriveSimulationRunningState(DeepDriveSimulationStateMachine &stateMachine)
	: DeepDriveSimulationStateBase(stateMachine, "Running")
{
}

DeepDriveSimulationRunningState::~DeepDriveSimulationRunningState()
{
}


void DeepDriveSimulationRunningState::enter(ADeepDriveSimulation &deepDriveSim)
{

}

void DeepDriveSimulationRunningState::update(ADeepDriveSimulation &deepDriveSim, float dT)
{
	if (deepDriveSim.m_CaptureProxy)
		deepDriveSim.m_CaptureProxy->update(dT);

	if (deepDriveSim.m_ServerProxy)
		deepDriveSim.m_ServerProxy->update(dT);
}

void DeepDriveSimulationRunningState::exit(ADeepDriveSimulation &deepDriveSim)
{
}