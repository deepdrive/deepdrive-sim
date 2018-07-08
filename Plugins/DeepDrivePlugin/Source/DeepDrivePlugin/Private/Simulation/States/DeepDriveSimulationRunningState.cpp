
#include "DeepDrivePluginPrivatePCH.h"
#include "Private/Simulation/States/DeepDriveSimulationRunningState.h"
#include "Public/Simulation/DeepDriveSimulation.h"
#include "Public/Simulation/DeepDriveSimulationServerProxy.h"
#include "Public/Simulation/DeepDriveSimulationCaptureProxy.h"

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