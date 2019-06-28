
#include "DeepDrivePluginPrivatePCH.h"
#include "Private/Simulation/States/DeepDriveSimulationInitializeState.h"
#include "Public/Simulation/DeepDriveSimulation.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoadNetworkComponent.h"

DeepDriveSimulationInitializeState::DeepDriveSimulationInitializeState(DeepDriveSimulationStateMachine &stateMachine, bool scenarioMode)
	: DeepDriveSimulationStateBase(stateMachine, "Initialize")
	,	m_ScenarioMode(scenarioMode)
{
}

DeepDriveSimulationInitializeState::~DeepDriveSimulationInitializeState()
{
}


void DeepDriveSimulationInitializeState::enter(ADeepDriveSimulation &deepDriveSim)
{
	deepDriveSim.RegisterRandomStream("AgentPlacement", false);
	deepDriveSim.RoadNetwork->Initialize(deepDriveSim);
}

void DeepDriveSimulationInitializeState::update(ADeepDriveSimulation &deepDriveSim, float dT)
{
	if (m_ScenarioMode == false)
	{
		deepDriveSim.initializeAgents();
		m_StateMachine.setNextState("Running");
	}
}

void DeepDriveSimulationInitializeState::exit(ADeepDriveSimulation &deepDriveSim)
{
	deepDriveSim.OnSimulationReady();
}