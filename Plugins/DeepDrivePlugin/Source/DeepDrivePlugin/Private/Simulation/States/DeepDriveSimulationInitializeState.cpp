
#include "DeepDrivePluginPrivatePCH.h"
#include "Private/Simulation/States/DeepDriveSimulationInitializeState.h"
#include "Public/Simulation/DeepDriveSimulation.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoadNetworkComponent.h"

DeepDriveSimulationInitializeState::DeepDriveSimulationInitializeState(DeepDriveSimulationStateMachine &stateMachine)
	: DeepDriveSimulationStateBase(stateMachine, "Initialize")
{
}

DeepDriveSimulationInitializeState::~DeepDriveSimulationInitializeState()
{
}


void DeepDriveSimulationInitializeState::enter(ADeepDriveSimulation &deepDriveSim)
{

}

void DeepDriveSimulationInitializeState::update(ADeepDriveSimulation &deepDriveSim, float dT)
{
	/*
		- register random streams
		- call blueprint initialize
		- spawn agents
	*/

	deepDriveSim.RegisterRandomStream("AgentPlacement", false);
	deepDriveSim.RoadNetwork->Initialize();
	deepDriveSim.initializeAgents();

	m_StateMachine.setNextState("Running");
}

void DeepDriveSimulationInitializeState::exit(ADeepDriveSimulation &deepDriveSim)
{
	deepDriveSim.OnSimulationReady();
}