

#include "DeepDrivePluginPrivatePCH.h"
#include "DeepDriveAgentLocalAIController.h"
#include "Public/Simulation/Misc/DeepDriveSplineTrack.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"
#include "Private/Simulation/Agent/Controllers/DeepDriveAgentSpeedController.h"
#include "Private/Simulation/Agent/Controllers/DeepDriveAgentSteeringController.h"

#include "Private/Simulation/Agent/Controllers/LocalAI/States/DeepDriveAgentCruiseState.h"
#include "Private/Simulation/Agent/Controllers/LocalAI/States/DeepDriveAgentPullOutState.h"
#include "Private/Simulation/Agent/Controllers/LocalAI/States/DeepDriveAgentPassingState.h"
#include "Private/Simulation/Agent/Controllers/LocalAI/States/DeepDriveAgentPullInState.h"

#include "Private/Simulation/Agent/Controllers/LocalAI/States/DeepDriveAgentAbortOvertakingState.h"

DEFINE_LOG_CATEGORY(LogDeepDriveAgentLocalAIController);

ADeepDriveAgentLocalAIController::ADeepDriveAgentLocalAIController()
{
	m_ControllerName = "Local AI Controller";
	m_isGameDriving = true;
}


bool ADeepDriveAgentLocalAIController::Activate(ADeepDriveAgent &agent)
{
	if (m_Track == 0)
	{
		TArray<AActor*> tracks;
		UGameplayStatics::GetAllActorsOfClass(GetWorld(), ADeepDriveSplineTrack::StaticClass(), tracks);
		for(auto &t : tracks)
		{
			if(t->ActorHasTag(FName("MainTrack")))
			{
				UE_LOG(LogDeepDriveAgentLocalAIController, Log, TEXT("LogDeepDriveAgentLocalAIController::Activate Found main track") );
				m_Track = Cast<ADeepDriveSplineTrack>(t);
				break;
			}
		}
	}

	if (m_Track)
	{
		m_SpeedController = new DeepDriveAgentSpeedController(m_Configuration.PIDThrottle, m_Configuration.PIDBrake);
		m_SpeedController->initialize(agent, *m_Track, m_Configuration.SafetyDistanceFactor);

		m_SteeringController = new DeepDriveAgentSteeringController(m_Configuration.PIDSteering);
		m_SteeringController->initialize(agent, *m_Track);

		if (m_SpeedController && m_SteeringController)
		{
			m_Spline = m_Track->GetSpline();
			resetAgentPosOnSpline(agent);

			UE_LOG(LogDeepDriveAgentLocalAIController, Log, TEXT("ADeepDriveAgentSplineController::Activate Successfully initialized"));
		}
	}
	else
		UE_LOG(LogDeepDriveAgentLocalAIController, Error, TEXT("ADeepDriveAgentSplineController::Activate Didn't find spline"));


	const bool activated = m_Spline != 0 && ADeepDriveAgentControllerBase::Activate(agent);

	if (activated)
	{
		m_StateMachineCtx = new DeepDriveAgentLocalAIStateMachineContext(*this, agent, *m_SpeedController, *m_SteeringController, m_Configuration);

		m_StateMachine.registerState(new DeepDriveAgentCruiseState(m_StateMachine));
		m_StateMachine.registerState(new DeepDriveAgentPullOutState(m_StateMachine));
		m_StateMachine.registerState(new DeepDriveAgentPassingState(m_StateMachine));
		m_StateMachine.registerState(new DeepDriveAgentPullInState(m_StateMachine));
		m_StateMachine.registerState(new DeepDriveAgentAbortOvertakingState(m_StateMachine));

		m_StateMachine.setNextState("Cruise");
	}

	return activated;
}

bool ADeepDriveAgentLocalAIController::ResetAgent()
{
	if(m_Track && m_Agent)
	{
		UE_LOG(LogDeepDriveAgentLocalAIController, Log, TEXT("Reset Agent") );
		m_Agent->reset();
		resetAgentPosOnSpline(*m_Agent);
		return true;
	}
	return false;
}


void ADeepDriveAgentLocalAIController::Configure(const FDeepDriveLocalAIControllerConfiguration &Configuration, int32 StartPositionSlot)
{
	m_Configuration = Configuration;

	m_DesiredSpeed = FMath::RandRange(Configuration.SpeedRange.X, Configuration.SpeedRange.Y);
	m_Track = Configuration.Track;
	m_StartDistance = Configuration.StartDistances[StartPositionSlot];

	UE_LOG(LogDeepDriveAgentLocalAIController, Log, TEXT("Configure %c"), m_Configuration.OvertakingEnabled ? 'T' : 'F');

}


void ADeepDriveAgentLocalAIController::Tick( float DeltaSeconds )
{
	if (m_Agent)
	{
		if(m_StateMachineCtx)
			m_StateMachine.update(*m_StateMachineCtx, DeltaSeconds);
	}
}


void ADeepDriveAgentLocalAIController::resetAgentPosOnSpline(ADeepDriveAgent &agent)
{
	FVector agentLocation = m_StartDistance > 0.0f ? (m_Spline->GetLocationAtDistanceAlongSpline(m_StartDistance, ESplineCoordinateSpace::World) + FVector(0.0f, 0.0f, 200.0f)) : agent.GetActorLocation();
	float curDistanceOnSpline = getClosestDistanceOnSpline(agentLocation);
	FVector curPosOnSpline = m_Spline->GetLocationAtDistanceAlongSpline(curDistanceOnSpline, ESplineCoordinateSpace::World);
	curPosOnSpline.Z = agentLocation.Z + 50.0f;

	FQuat quat = m_Spline->GetQuaternionAtDistanceAlongSpline(curDistanceOnSpline, ESplineCoordinateSpace::World);

	FTransform transform(quat.Rotator(), curPosOnSpline, FVector(1.0f, 1.0f, 1.0f));

	agent.SetActorTransform(transform, false, 0, ETeleportType::TeleportPhysics);
}

float ADeepDriveAgentLocalAIController::getClosestDistanceOnSpline(const FVector &location)
{
	float distance = 0.0f;

	const float closestKey = m_Spline->FindInputKeyClosestToWorldLocation(location);

	const int32 index0 = floor(closestKey);
	const int32 index1 = floor(closestKey + 1.0f);

	const float dist0 = m_Spline->GetDistanceAlongSplineAtSplinePoint(index0);
	const float dist1 = m_Spline->GetDistanceAlongSplineAtSplinePoint(index1);


	return FMath::Lerp(dist0, dist1, closestKey - static_cast<float> (index0));
}

float ADeepDriveAgentLocalAIController::calculateOvertakingScore()
{
	float distanceToNextAgent = -1.0f;
	ADeepDriveAgent *nextAgent = m_Agent->getNextAgent(&distanceToNextAgent);

	//UE_LOG(LogDeepDriveAgentLocalAIController, Log, TEXT("calculateOvertakingScore: Agent %d dist %f"), m_Agent->getAgentId(), distanceToNextAgent);

	float score = -1.0f;

	if (distanceToNextAgent <= m_Configuration.OvertakingMinDistance)
	{
		score = 1.0f;
	}

	if (nextAgent->getSpeed() * 0.036f < m_Configuration.MinSpeedDifference)
		score -= 1.0f;

	float nextButOneDist = -1.0f;
	ADeepDriveAgent *nextButOne = nextAgent->getNextAgent(&nextButOneDist);
	if (nextButOne != m_Agent)
	{
		if (nextButOne && nextButOneDist < m_Configuration.GapBetweenAgents)
			score -= 1.0f;
	}
	return score;
}

float ADeepDriveAgentLocalAIController::calculateAbortOvertakingScore()
{
	float score = 0.0f;

	return score;
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
	if(m_Track->getNextAgent(nextAgent, nextButOne, nextButOneDist))
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


void ADeepDriveAgentLocalAIController::OnCheckpointReached()
{
}

void ADeepDriveAgentLocalAIController::OnDebugTrigger()
{
	if (m_isPaused)
	{
		UGameplayStatics::SetGamePaused(GetWorld(), false);
		m_isPaused = false;
	}
	else
	{
		FVector agentLocation = m_Agent->GetActorLocation();
		const float curKey = m_Spline->FindInputKeyClosestToWorldLocation(agentLocation);

		UE_LOG(LogDeepDriveAgentLocalAIController, Log, TEXT("Current key: %f"), curKey);

		UGameplayStatics::SetGamePaused(GetWorld(), true);
		m_isPaused = true;
	}
}
