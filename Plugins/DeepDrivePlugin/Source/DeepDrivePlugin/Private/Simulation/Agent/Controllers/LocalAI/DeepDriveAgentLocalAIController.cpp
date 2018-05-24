

#include "DeepDrivePluginPrivatePCH.h"
#include "DeepDriveAgentLocalAIController.h"
#include "Public/Simulation/Misc/DeepDriveSplineTrack.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"
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

	if (Track)
	{
		m_SplineDrivingCtrl = new DeepDriveAgentSplineDrivingCtrl(m_Configuration.PIDSteering, m_Configuration.PIDThrottle, m_Configuration.PIDBrake);
		if (m_SplineDrivingCtrl)
		{
			m_SplineDrivingCtrl->initialize(agent, Track);

			m_Spline = Track->GetSpline();
			resetAgentPosOnSpline(agent);

			UE_LOG(LogDeepDriveAgentLocalAIController, Log, TEXT("ADeepDriveAgentSplineController::Activate Successfully initialized"));
		}
	}
	else
		UE_LOG(LogDeepDriveAgentLocalAIController, Error, TEXT("ADeepDriveAgentSplineController::Activate Didn't find spline"));


	const bool activated = m_Spline != 0 && m_SplineDrivingCtrl != 0 && ADeepDriveAgentControllerBase::Activate(agent);

	if (activated)
	{
		m_SplineDrivingCtrl->setSafetyDistanceFactor(m_Configuration.SafetyDistanceFactor);
		m_SplineDrivingCtrl->setBrakingDistanceRange(m_Configuration.BrakingDistanceRange);

		m_StateMachineCtx = new DeepDriveAgentLocalAIStateMachineContext(*this, agent, *m_SplineDrivingCtrl, m_Configuration);

		m_StateMachine.registerState(new DeepDriveAgentCruisingState(m_StateMachine));
		m_StateMachine.registerState(new DeepDriveAgentBeginOvertakingState(m_StateMachine));
		m_StateMachine.registerState(new DeepDriveAgentOvertakingState(m_StateMachine));
		m_StateMachine.registerState(new DeepDriveAgentFinishOvertakingState(m_StateMachine));
		m_StateMachine.registerState(new DeepDriveAgentAbortOvertakingState(m_StateMachine));

		m_StateMachine.setNextState("Cruising");
	}

	return activated;
}


void ADeepDriveAgentLocalAIController::Configure(const FDeepDriveLocalAIControllerConfiguration &Configuration, int32 StartPositionSlot)
{
	m_Configuration = Configuration;

	DesiredSpeed = FMath::RandRange(Configuration.SpeedRange.X, Configuration.SpeedRange.Y);
	Track = Configuration.Track;
	StartDistance = Configuration.StartDistances[StartPositionSlot];

	UE_LOG(LogDeepDriveAgentLocalAIController, Log, TEXT("Configure %c"), m_Configuration.OvertakingEnabled ? 'T' : 'F');

}


void ADeepDriveAgentLocalAIController::Tick( float DeltaSeconds )
{
	if (m_Agent && m_SplineDrivingCtrl)
	{
		think(DeltaSeconds);

		if(m_StateMachineCtx)
			m_StateMachine.update(*m_StateMachineCtx, DeltaSeconds);
	}
}

void ADeepDriveAgentLocalAIController::think(float dT)
{
	if(Track->getNextAgent(*m_Agent, m_StateMachineCtx->next_agent, m_StateMachineCtx->distance_to_next_agent) == false)
	{
		m_StateMachineCtx->next_agent = 0;
		m_StateMachineCtx->distance_to_next_agent = -1.0f;
	}

	if(m_Configuration.OvertakingEnabled && m_StateMachineCtx->next_agent)
	{
		m_StateMachineCtx->overtaking_score = calculateOvertakingScore(*m_StateMachineCtx->next_agent, m_StateMachineCtx->distance_to_next_agent);
		UE_LOG(LogDeepDriveAgentLocalAIController, Log, TEXT("Overtaking score %f"), m_StateMachineCtx->overtaking_score);
	}
	else
	{
		m_StateMachineCtx->overtaking_score = -1.0f;
		m_StateMachineCtx->overtaking_in_progess = false;
	}

}

float ADeepDriveAgentLocalAIController::calculateOvertakingScore(ADeepDriveAgent &nextAgent, float distanceToNextAgent)
{
	float score = -1.0f;

	if	(	m_StateMachineCtx->overtaking_in_progess == false
		&&	distanceToNextAgent <= m_Configuration.OvertakingMinDistance
		)
	{
		score = 1.0f;
	}

	if(nextAgent.getSpeed() * 0.036f < m_Configuration.MinSpeedDifference)
		score -= 1.0f;

	ADeepDriveAgent *nextButOne = 0;
	float nextButOneDist = -1.0f;
	if(Track->getNextAgent(nextAgent, nextButOne, nextButOneDist))
	{
		if(nextButOne && nextButOneDist < m_Configuration.GapBetweenAgents)
			score -= 1.0f;
	}

/*
	ADeepDriveAgent *nextOpposing = 0;
	float nextOpposingDist = -1.0f;
	if(Track->getNextAgent(nextAgent, nextButOne, nextButOneDist))
	{
		if(nextOpposing && nextButOneDist > m_Configuration.GapBetweenAgents)
			score += 0.25f;
		else
			score = -1.0f;
	}
	else
		score += 1.0f;	
*/
	return score;
}


/*

- extend state machine context

struct DeepDriveAgentLocalAIStateMachineContext
{

	ADeepDriveAgent										*next_agent = 0;
	float												distance_to_next_agent = -1.0f;

	float												overtaking_score = 0.0f;
	bool												overtaking_in_progess = false;

	float												side_offset = 0.0f;

};

Think:

- every x seconds check for next agent (0.25)
- if overtaking enabled, every x seconds calculate overtaking score


float calculateOvertakingScore()
{
	- next agent must not be overtaking
	- distance to next agent must be smaller than OvertakingMinDistance (applies only when overtaking_in_progess == false)
	- speed difference to next agent must be greater than MinSpeedDifference
	- distance between next agent and its next agent must be greater than GapBetweenAgents
	- estimated remaning overtaking distance must be smaller than distance to next agent on opposing lane
}


*/
