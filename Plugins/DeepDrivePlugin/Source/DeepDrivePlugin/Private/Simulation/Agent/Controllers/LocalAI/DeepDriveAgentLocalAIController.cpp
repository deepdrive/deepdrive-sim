

#include "DeepDrivePluginPrivatePCH.h"
#include "DeepDriveAgentLocalAIController.h"
#include "Public/Simulation/Misc/DeepDriveSplineTrack.h"

#include "Simulation/Agent/Controllers/LocalAI/DeepDriveAgentLocalAIStateBase.h"

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

	m_StateMachineCtx = new DeepDriveAgentLocalAIStateMachineContext(*this, agent);


	return ADeepDriveAgentSplineController::Activate(agent);
}


void ADeepDriveAgentLocalAIController::Tick( float DeltaSeconds )
{
	if (m_Agent && m_SplineDrivingCtrl)
	{
		if(m_StateMachineCtx)
			m_StateMachine.update(*m_StateMachineCtx, DeltaSeconds);

		m_SplineDrivingCtrl->update(DeltaSeconds, DesiredSpeed, 0.0f);
	}
}
