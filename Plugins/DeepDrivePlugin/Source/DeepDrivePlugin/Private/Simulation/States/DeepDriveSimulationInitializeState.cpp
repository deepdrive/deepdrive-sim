
#include "DeepDrivePluginPrivatePCH.h"
#include "Private/Simulation/States/DeepDriveSimulationInitializeState.h"
#include "Public/Simulation/DeepDriveSimulation.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoadNetworkComponent.h"

DEFINE_LOG_CATEGORY(LogDeepDriveSimulationState);

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

	UE_LOG(LogDeepDriveSimulationState, Log, TEXT("Initialize: ScenarioMode %c"), m_ScenarioMode ? 'T' : 'F' );
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
