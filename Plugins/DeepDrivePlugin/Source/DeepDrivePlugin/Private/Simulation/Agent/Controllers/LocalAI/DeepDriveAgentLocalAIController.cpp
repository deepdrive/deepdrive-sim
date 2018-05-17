

#include "DeepDrivePluginPrivatePCH.h"
#include "DeepDriveAgentLocalAIController.h"
#include "Public/Simulation/Misc/DeepDriveSplineTrack.h"
#include "Private/Simulation/Agent/Controllers/LocalAI/States/DeepDriveAgentCruisingState.h"
#include "Private/Simulation/Agent/Controllers/DeepDriveAgentSplineDrivingCtrl.h"

#include "Private/Simulation/Agent/Controllers/LocalAI/States/DeepDriveAgentCruisingState.h"
#include "Private/Simulation/Agent/Controllers/LocalAI/States/DeepDriveAgentBeginOvertakingState.h"
#include "Private/Simulation/Agent/Controllers/LocalAI/States/DeepDriveAgentOvertakingState.h"
#include "Private/Simulation/Agent/Controllers/LocalAI/States/DeepDriveAgentFinishOvertakingState.h"
#include "Private/Simulation/Agent/Controllers/LocalAI/States/DeepDriveAgentAbortOvertakingState.h"

DEFINE_LOG_CATEGORY(LogDeepDriveAgentLocalAIController);

ADeepDriveAgentLocalAIController::ADeepDriveAgentLocalAIController()
{
	m_ControllerName = "Local AI Controller";
	m_isGameDriving = true;
}


bool ADeepDriveAgentLocalAIController::Activate(ADeepDriveAgent &agent)
{
	if (Track == 0)
	{
		TArray<AActor*> tracks;
		UGameplayStatics::GetAllActorsOfClass(GetWorld(), ADeepDriveSplineTrack::StaticClass(), tracks);
		for(auto &t : tracks)
		{
			if(t->ActorHasTag(FName("MainTrack")))
			{
				UE_LOG(LogDeepDriveAgentLocalAIController, Log, TEXT("LogDeepDriveAgentLocalAIController::Activate Found main track") );
				Track = Cast<ADeepDriveSplineTrack>(t);
				break;
			}
		}
	}

	const bool activated = ADeepDriveAgentSplineController::Activate(agent);

	if (activated)
	{
		m_StateMachineCtx = new DeepDriveAgentLocalAIStateMachineContext(*this, agent, *m_SplineDrivingCtrl);

		m_StateMachine.registerState(new DeepDriveAgentCruisingState(m_StateMachine));
		m_StateMachine.registerState(new DeepDriveAgentBeginOvertakingState(m_StateMachine));
		m_StateMachine.registerState(new DeepDriveAgentOvertakingState(m_StateMachine));
		m_StateMachine.registerState(new DeepDriveAgentFinishOvertakingState(m_StateMachine));
		m_StateMachine.registerState(new DeepDriveAgentAbortOvertakingState(m_StateMachine));

		m_StateMachine.setNextState("Cruising");
	}

	return activated;
}


void ADeepDriveAgentLocalAIController::Tick( float DeltaSeconds )
{
	if (m_Agent && m_SplineDrivingCtrl)
	{
		if(m_StateMachineCtx)
			m_StateMachine.update(*m_StateMachineCtx, DeltaSeconds);

	}
}
