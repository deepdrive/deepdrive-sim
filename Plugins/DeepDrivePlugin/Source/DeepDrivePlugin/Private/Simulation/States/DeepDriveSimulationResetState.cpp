
#include "DeepDrivePluginPrivatePCH.h"
#include "Private/Simulation/States/DeepDriveSimulationResetState.h"
#include "Public/Simulation/DeepDriveSimulation.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"
#include "Public/Simulation/Agent/DeepDriveAgentControllerBase.h"
#include "Public/Simulation/Misc/DeepDriveSplineTrack.h"
#include "Public/Simulation/Misc/DeepDriveRandomStream.h"

DEFINE_LOG_CATEGORY(LogDeepDriveSimulationResetState);

DeepDriveSimulationResetState::DeepDriveSimulationResetState(DeepDriveSimulationStateMachine &stateMachine, UWorld *world)
	:	DeepDriveSimulationStateBase(stateMachine, "Reset")
	,	m_World(world)
{
}

DeepDriveSimulationResetState::~DeepDriveSimulationResetState()
{
}


void DeepDriveSimulationResetState::enter(ADeepDriveSimulation &deepDriveSim)
{

}

void DeepDriveSimulationResetState::update(ADeepDriveSimulation &deepDriveSim, float dT)
{
	UE_LOG(LogDeepDriveSimulation, Log, TEXT("Reset simulation") );
	for (auto &rs : deepDriveSim.RandomStreams)
	{
		if (rs.Value.ReSeedOnReset && rs.Value.getRandomStream())
			rs.Value.getRandomStream()->initialize(deepDriveSim.Seed);
	}

	TArray< AActor*> actors;
	UGameplayStatics::GetAllActorsOfClass(m_World, ADeepDriveSplineTrack::StaticClass(), actors);
	for (auto actor : actors)
	{
		ADeepDriveSplineTrack *track = Cast<ADeepDriveSplineTrack>(actor);
		if (track)
			track->resetRandomSlots();
	}

	deepDriveSim.removeOneOffAgents();

	if(m_ActivateAdditionalAgents)
	{
		if(deepDriveSim.hasAdditionalAgents() == false)
		{
			UE_LOG(LogDeepDriveSimulation, Log, TEXT("Spawning additional agents") );
			deepDriveSim.spawnAdditionalAgents();
		}
	}
	else
	{
		if(deepDriveSim.hasAdditionalAgents())
		{
			UE_LOG(LogDeepDriveSimulation, Log, TEXT("Removing additional agents") );
			deepDriveSim.removeAgents(false);
		}
	}

	UE_LOG(LogDeepDriveSimulation, Log, TEXT("Reseting agents %d"), deepDriveSim.m_Agents.Num() );
	for (auto &agent : deepDriveSim.m_Agents)
	{
		ADeepDriveAgentControllerBase *controller = Cast<ADeepDriveAgentControllerBase>(agent->GetController());
		if (controller)
			controller->ResetAgent();

	}

	m_StateMachine.setNextState("Running");
}

void DeepDriveSimulationResetState::exit(ADeepDriveSimulation &deepDriveSim)
{
}