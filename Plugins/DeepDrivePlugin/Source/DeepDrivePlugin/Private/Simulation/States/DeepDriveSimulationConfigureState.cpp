
#include "DeepDrivePluginPrivatePCH.h"
#include "Private/Simulation/States/DeepDriveSimulationConfigureState.h"
#include "Public/Simulation/DeepDriveSimulation.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"
#include "Public/Simulation/Agent/DeepDriveAgentControllerBase.h"
#include "Public/Simulation/Misc/DeepDriveSplineTrack.h"
#include "Public/Simulation/Misc/DeepDriveRandomStream.h"

DEFINE_LOG_CATEGORY(LogDeepDriveSimulationConfigureState);

DeepDriveSimulationConfigureState::DeepDriveSimulationConfigureState(DeepDriveSimulationStateMachine &stateMachine, UWorld *world)
	:	DeepDriveSimulationStateBase(stateMachine, "Configure")
	,	m_World(world)
{
}

DeepDriveSimulationConfigureState::~DeepDriveSimulationConfigureState()
{
}


void DeepDriveSimulationConfigureState::enter(ADeepDriveSimulation &deepDriveSim)
{

}

void DeepDriveSimulationConfigureState::update(ADeepDriveSimulation &deepDriveSim, float dT)
{
	UE_LOG(LogDeepDriveSimulation, Log, TEXT("Configure simulation") );

	for (auto &rs : deepDriveSim.RandomStreams)
	{
		if (rs.Value.ReSeedOnReset && rs.Value.getRandomStream())
			rs.Value.getRandomStream()->initialize(deepDriveSim.Seed);
	}

	deepDriveSim.removeOneOffAgents();
	deepDriveSim.removeAgents(true);

	UE_LOG(LogDeepDriveSimulation, Log, TEXT("Spawning ego agent"));

	deepDriveSim.m_curAgent = deepDriveSim.spawnAgent(m_Configuration.EgoAgent, m_Configuration.EgoAgent.IsRemotelyControlled);
	deepDriveSim.OnCurrentAgentChanged(deepDriveSim.m_curAgent);

	deepDriveSim.m_curAgentController = Cast<ADeepDriveAgentControllerBase>(deepDriveSim.m_curAgent->GetController());
	deepDriveSim.SelectCamera(EDeepDriveAgentCameraType::CHASE_CAMERA);

	UE_LOG(LogDeepDriveSimulation, Log, TEXT("Spawning additional agents %d"), m_Configuration.Agents.Num());
	for (auto &agentCfg : m_Configuration.Agents)
	{
		ADeepDriveAgent *agent = deepDriveSim.spawnAgent(agentCfg, false);
	}

	m_StateMachine.setNextState("Running");

}

void DeepDriveSimulationConfigureState::exit(ADeepDriveSimulation &deepDriveSim)
{
}