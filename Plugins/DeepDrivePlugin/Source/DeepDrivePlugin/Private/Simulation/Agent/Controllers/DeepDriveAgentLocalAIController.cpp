

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
#include "Private/Simulation/Agent/Controllers/LocalAI/States/DeepDriveAgentPullBackInState.h"

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
		m_Track->registerAgent(agent, m_Track->GetSpline()->FindInputKeyClosestToWorldLocation(agent.GetActorLocation()));

		m_SpeedController = new DeepDriveAgentSpeedController(m_Configuration.PIDThrottle, m_Configuration.PIDBrake);
		m_SpeedController->initialize(agent, *m_Track, m_Configuration.SafetyDistanceFactor);

		m_SteeringController = new DeepDriveAgentSteeringController(m_Configuration.PIDSteering);
		m_SteeringController->initialize(agent, *m_Track);

		m_OppositeTrack = m_Track->OppositeTrack;

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
		m_StateMachine.registerState(new DeepDriveAgentPullBackInState(m_StateMachine));
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
	m_SafetyDistanceFactor = Configuration.SafetyDistanceFactor;
}


void ADeepDriveAgentLocalAIController::Tick( float DeltaSeconds )
{
	if (m_Agent && m_Track)
	{
		FVector curAgentLocation = m_Agent->GetActorLocation();
		m_Track->setBaseLocation(curAgentLocation);

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

#if 0
float ADeepDriveAgentLocalAIController::calculateOvertakingScore()
{
	float score = -1.0f;

	float distanceToNextAgent = -1.0f;
	ADeepDriveAgent *nextAgent = m_Agent->getNextAgent(&distanceToNextAgent);

	if(nextAgent)
	{
		//UE_LOG(LogDeepDriveAgentLocalAIController, Log, TEXT("calculateOvertakingScore: Agent %d dist %f"), m_Agent->getAgentId(), distanceToNextAgent);

		if (distanceToNextAgent <= m_Configuration.MinPullOutDistance)
		{
			score = 1.0f;
		}

		if (nextAgent->getSpeedKmh() < m_Configuration.MinSpeedDifference)
			score -= 1.0f;

		float nextButOneDist = -1.0f;
		ADeepDriveAgent *nextButOne = nextAgent->getNextAgent(&nextButOneDist);
		if (nextButOne != m_Agent)
		{
			if (nextButOne && nextButOneDist < m_Configuration.GapBetweenAgents)
				score -= 1.0f;
		}

	}

	return score;
}

float ADeepDriveAgentLocalAIController::calculateOvertakingScore(int32 maxAgentsToOvertake, float overtakingSpeed, ADeepDriveAgent* &finalAgent)
{
	float score = -1.0f;
	finalAgent = 0;

	float distanceToNextAgent = -1.0f;
	ADeepDriveAgent *nextAgent = m_Agent->getNextAgent(&distanceToNextAgent);
	if (nextAgent && distanceToNextAgent <= m_Configuration.MinPullOutDistance)
	{
		score = 1.0f;
		while(maxAgentsToOvertake > 0 && nextAgent)
		{
			const float speedDiff = (overtakingSpeed - nextAgent->getSpeedKmh());
			if ( speedDiff < m_Configuration.MinSpeedDifference)
			{
				break;
			}

			float nextButOneDist = -1.0f;
			ADeepDriveAgent *nextButOne = nextAgent->getNextAgent(&nextButOneDist);

			if(nextButOne == 0 || nextButOneDist > m_Configuration.GapBetweenAgents)
				finalAgent = nextAgent;

			if	(	nextButOne == m_Agent
				||	nextButOneDist > 3.0f * m_Configuration.GapBetweenAgents
				)
				break;

			nextAgent = nextAgent->getNextAgent(&distanceToNextAgent);
			--maxAgentsToOvertake;
			score *= 0.9f;
		}
	}

	return finalAgent ? score : -1.0f;
}

float ADeepDriveAgentLocalAIController::calculateAbortOvertakingScore()
{
	float score = 0.0f;

	return score;
}

bool ADeepDriveAgentLocalAIController::hasPassed(ADeepDriveAgent *other, float minDistance)
{
	bool hasPassed = false;

	float dist2Prev = -1.0f;
	ADeepDriveAgent *prevAgent = m_Agent->getPrevAgent(-1.0f, &dist2Prev);

	if(prevAgent == other)
	{
		FVector dir = m_Agent->GetActorLocation() - prevAgent->GetActorLocation();
		dir.Normalize();

		if(FVector::DotProduct(prevAgent->GetActorForwardVector(), dir) > 0.5f)
		{
			hasPassed = dist2Prev >= minDistance;
		}
	}

	return hasPassed;
}

#endif

float ADeepDriveAgentLocalAIController::getPassedDistance(ADeepDriveAgent *other)
{
	float distance = 0.0f;
	ADeepDriveAgent *prevAgent = m_Agent->getPrevAgent(-1.0f, &distance);
	if (prevAgent == other)
	{
		FVector dir = m_Agent->GetActorLocation() - prevAgent->GetActorLocation();
		dir.Normalize();

		if (FVector::DotProduct(prevAgent->GetActorForwardVector(), dir) > 0.0f)
		{
			return distance;
		}
	}

	return -1.0f;
}

float ADeepDriveAgentLocalAIController::isOppositeTrackClear(ADeepDriveAgent &nextAgent, float distanceToNextAgent, float speedDifference, float overtakingSpeed, bool considerDuration)
{
	float res = 1.0f;

	if(m_OppositeTrack)
	{
		// calculate pure overtaking distance
		const float pureOvertakingDistance = distanceToNextAgent + m_Configuration.MinPullInDistance + nextAgent.getFrontBumperDistance() + m_Agent->getBackBumperDistance();
		// calcualte time nased on speed difference
		const float overtakingDuration = pureOvertakingDistance / (speedDifference * 100.0f * 1000.0f / 3600.0f) ;
		// calculate distance covered in that time based on theoretically overtaking speed
		float overtakingDistance = overtakingDuration * overtakingSpeed * 100.0f * 1000.0f / 3600.0f;

		ADeepDriveAgent *prevAgent;
		float distanceToPrev = 0.0f;

		m_OppositeTrack->getPreviousAgent(m_Agent->GetActorLocation(), prevAgent, distanceToPrev);
		if(prevAgent)
		{
			if(considerDuration)
			{
				const float coveredDist = overtakingDuration * prevAgent->getSpeed();
				overtakingDistance += coveredDist;
			}

			const float d = (prevAgent->GetActorLocation() - m_Agent->GetActorLocation()).Size();

			res = distanceToPrev / overtakingDistance;
			//UE_LOG(LogDeepDriveAgentLocalAIController, Log, TEXT("Agent %s Spd %4.1f|%4.1f|%4.1f AirDist %f dtn %f dtp %f OvrTkDist %f|%f Duration %f Otc %f"), *(prevAgent->GetName()), prevAgent->getSpeed(), overtakingSpeed, speedDifference, d, distanceToNextAgent, distanceToPrev, pureOvertakingDistance, overtakingDistance, overtakingDuration, res);
		}
	}
	return res;
}

float ADeepDriveAgentLocalAIController::computeOppositeTrackClearance(float overtakingDistance, float speedDifference, float overtakingSpeed, bool considerDuration)
{
	float res = 1.0f;

	if(m_OppositeTrack)
	{
		// calcualte time nased on speed difference
		const float overtakingDuration = overtakingDistance / (speedDifference * 100.0f * 1000.0f / 3600.0f) ;
		// calculate distance covered in that time based on theoretically overtaking speed
		overtakingDistance = overtakingDuration * overtakingSpeed * 100.0f * 1000.0f / 3600.0f;

		ADeepDriveAgent *prevAgent;
		float distanceToPrev = 0.0f;

		m_OppositeTrack->getPreviousAgent(m_Agent->GetActorLocation(), prevAgent, distanceToPrev);
		if(prevAgent)
		{
			if(considerDuration)
			{
				const float coveredDist = overtakingDuration * prevAgent->getSpeed();
				overtakingDistance += coveredDist;
			}

			const float d = (prevAgent->GetActorLocation() - m_Agent->GetActorLocation()).Size();

			res = distanceToPrev / overtakingDistance;
			UE_LOG(LogDeepDriveAgentLocalAIController, Log, TEXT("Agent %s Spd %4.1f|%4.1f|%4.1f AirDist %f dtp %f OvrTkDist %f Duration %f Otc %f"), *(prevAgent->GetName()), prevAgent->getSpeed(), overtakingSpeed, speedDifference, d, distanceToPrev,overtakingDistance, overtakingDuration, res);
		}
	}
	return res;
}

#if 0
bool ADeepDriveAgentLocalAIController::isOppositeTrackClear(float distance, float duration)
{
	bool res = true;

	if(m_OppositeTrack)
	{
		ADeepDriveAgent *prevAgent;
		float distanceToPrev = 0.0f;

		m_OppositeTrack->getPreviousAgent(m_Agent->GetActorLocation(), prevAgent, distanceToPrev);
		if(prevAgent)
		{
			if(duration > 0.0f)
			{
				const float coveredDist = duration * prevAgent->getSpeed();
				distance += coveredDist;
			}
			res = distance < distanceToPrev;
		}
	}
	return res;
}

float ADeepDriveAgentLocalAIController::calculateSafetyDistance(float *curDistance)
{
	float safetyDistance = -1.0f;
	ADeepDriveAgent *nextAgent = m_Agent->getNextAgent(curDistance);
	if (nextAgent)
	{
		const float curSpeed = m_Agent->getSpeed();
		safetyDistance = m_SafetyDistanceFactor * curSpeed * curSpeed / (2.0f * m_BrakingDeceleration);
	}

	return safetyDistance;
}
#endif

float ADeepDriveAgentLocalAIController::calculateSafetyDistance()
{
	const float curSpeed = m_Agent->getSpeed();
	const float safetyDistance = m_SafetyDistanceFactor * curSpeed * curSpeed / (2.0f * m_BrakingDeceleration);

	return safetyDistance;
}

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
