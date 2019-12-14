

#include "DeepDrivePluginPrivatePCH.h"
#include "DeepDriveAgentLocalAIController.h"
#include "Public/Simulation/Misc/DeepDriveSplineTrack.h"
#include "Public/Simulation/Misc/DeepDriveRandomStream.h"
#include "Public/Simulation/DeepDriveSimulation.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"
#include "Private/Simulation/Agent/Controllers/DeepDriveAgentSpeedController.h"
#include "Private/Simulation/Agent/Controllers/DeepDriveAgentSteeringController.h"

#include "Private/Simulation/Agent/Controllers/LocalAI/States/DeepDriveAgentCruiseState.h"
#include "Private/Simulation/Agent/Controllers/LocalAI/States/DeepDriveAgentPullOutState.h"
#include "Private/Simulation/Agent/Controllers/LocalAI/States/DeepDriveAgentPassingState.h"
#include "Private/Simulation/Agent/Controllers/LocalAI/States/DeepDriveAgentPullInState.h"
#include "Private/Simulation/Agent/Controllers/LocalAI/States/DeepDriveAgentPullBackInState.h"

#include "Private/Simulation/Agent/Controllers/LocalAI/States/DeepDriveAgentAbortOvertakingState.h"

#include "WheeledVehicleMovementComponent.h"

DEFINE_LOG_CATEGORY(LogDeepDriveAgentLocalAIController);

ADeepDriveAgentLocalAIController::ADeepDriveAgentLocalAIController()
	:	ADeepDriveAgentControllerBase()
{
	m_ControllerName = "Local AI Controller";

	//	for testing purposes
	m_isCollisionEnabled = true;
}

bool ADeepDriveAgentLocalAIController::Activate(ADeepDriveAgent &agent, bool keepPosition)
{
	bool activated = false;

	if (keepPosition || initAgentOnTrack(agent))
	{
		m_SpeedController = new DeepDriveAgentSpeedController(m_Configuration.PIDThrottle, m_Configuration.PIDBrake);
		m_SpeedController->initialize(agent, *m_Track, m_Configuration.SafetyDistanceFactor);

		m_SteeringController = new DeepDriveAgentSteeringController(m_Configuration.PIDSteering);
		m_SteeringController->initialize(agent, *m_Track);

		m_OppositeTrack = m_Track->OppositeTrack;

		if (m_SpeedController && m_SteeringController)
		{
			if(m_StartDistance >= 0 && m_Track != 0)
			{
				activateController(agent);
				m_StateMachineCtx = new DeepDriveAgentLocalAIStateMachineContext(*this, agent, *m_SpeedController, *m_SteeringController, m_Configuration);

				if(m_StateMachineCtx)
				{
					m_StateMachine.registerState(new DeepDriveAgentCruiseState(m_StateMachine));
					m_StateMachine.registerState(new DeepDriveAgentPullOutState(m_StateMachine));
					m_StateMachine.registerState(new DeepDriveAgentPullBackInState(m_StateMachine));
					m_StateMachine.registerState(new DeepDriveAgentPassingState(m_StateMachine));
					m_StateMachine.registerState(new DeepDriveAgentPullInState(m_StateMachine));
					m_StateMachine.registerState(new DeepDriveAgentAbortOvertakingState(m_StateMachine));

					m_StateMachine.setNextState("Cruise");
					
					m_StateMachine.update(*m_StateMachineCtx, 0.0f);

					activated = true;
				}
			}
		}
	}
	else
		UE_LOG(LogDeepDriveAgentLocalAIController, Error, TEXT("ADeepDriveAgentSplineController::Activate Didn't find spline"));

	return activated;
}

bool ADeepDriveAgentLocalAIController::ResetAgent()
{
	if(m_Track && m_Agent)
	{
		UE_LOG(LogDeepDriveAgentLocalAIController, Log, TEXT("Reset Agent at %d"), m_StartPositionSlot);
		if(m_StartPositionSlot < 0)
			m_StartDistance = m_Track->getRandomDistanceAlongTrack(*m_DeepDriveSimulation->GetRandomStream(FName("AgentPlacement")));

		resetAgentPosOnSpline(*m_Agent, m_Track->GetSpline(), m_StartDistance);
		m_Agent->reset();
		m_StateMachine.setNextState("Cruise");
		m_SpeedController->reset();
		m_SteeringController->reset();
		return true;
	}
	return false;
}

void ADeepDriveAgentLocalAIController::SetSpeedRange(float MinSpeed, float MaxSpeed)
{
	m_DesiredSpeed = FMath::RandRange(MinSpeed, MaxSpeed);
}

void ADeepDriveAgentLocalAIController::Configure(const FDeepDriveLocalAIControllerConfiguration &Configuration, int32 StartPositionSlot, ADeepDriveSimulation* DeepDriveSimulation)
{
	m_Configuration = Configuration;
	m_DeepDriveSimulation = DeepDriveSimulation;

	m_DesiredSpeed = FMath::RandRange(Configuration.SpeedRange.X, Configuration.SpeedRange.Y);
	m_Track = Configuration.Track;
	m_StartPositionSlot = StartPositionSlot;
	m_StartDistance = StartPositionSlot >= 0 && StartPositionSlot < Configuration.StartDistances.Num() ? Configuration.StartDistances[StartPositionSlot] : -1.0f;
	m_SafetyDistanceFactor = Configuration.SafetyDistanceFactor;
}


void ADeepDriveAgentLocalAIController::Tick( float DeltaSeconds )
{
	if (m_InputTimer > 0.0f)
	{
		m_InputTimer -= DeltaSeconds;
	}
	else if (m_Agent && m_Track)
	{
		m_Agent->GetVehicleMovementComponent()->SetTargetGear(1, false);

		FVector curAgentLocation = m_Agent->GetActorLocation();
		m_Track->setBaseLocation(curAgentLocation);

		if(m_StateMachineCtx)
			m_StateMachine.update(*m_StateMachineCtx, DeltaSeconds);
	}
}

float ADeepDriveAgentLocalAIController::getPassedDistance(ADeepDriveAgent *other, float threshold)
{
	float distance = 0.0f;
	ADeepDriveAgent *prevAgent = m_Agent->getPrevAgent(-1.0f, &distance);
	if (prevAgent == other)
	{
		FVector dir = m_Agent->GetActorLocation() - prevAgent->GetActorLocation();
		dir.Normalize();

		if (FVector::DotProduct(prevAgent->GetActorForwardVector(), dir) > threshold)
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

float ADeepDriveAgentLocalAIController::computeOppositeTrackClearance(float overtakingSpeed, float lookAheadDuration)
{
	float res = 1.0f;

	if (m_OppositeTrack)
	{
		ADeepDriveAgent *prevAgent;
		float distanceToPrev = 0.0f;

		m_OppositeTrack->getPreviousAgent(m_Agent->GetActorLocation(), prevAgent, distanceToPrev);
		if (prevAgent)
		{
			float overtakingDistance = distanceToPrev;
			const float coveredDist = lookAheadDuration * prevAgent->getSpeed();
			overtakingDistance -= coveredDist;

			overtakingDistance = overtakingDistance - overtakingSpeed * 100.0f * lookAheadDuration;
			res = overtakingDistance / distanceToPrev;

			const float d = (prevAgent->GetActorLocation() - m_Agent->GetActorLocation()).Size();

			//UE_LOG(LogDeepDriveAgentLocalAIController, Log, TEXT("==> Agent %s Spd %4.1f|%4.1f AirDist %f dtp %f OvrTkDist %f => Otc %f"), *(prevAgent->GetName()), prevAgent->getSpeed(), overtakingSpeed, d, distanceToPrev, overtakingDistance, res);
		}
	}
	return res;
}

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
		const float curKey = m_Track->GetSpline()->FindInputKeyClosestToWorldLocation(agentLocation);

		UE_LOG(LogDeepDriveAgentLocalAIController, Log, TEXT("Current key: %f"), curKey);

		UGameplayStatics::SetGamePaused(GetWorld(), true);
		m_isPaused = true;
	}
}
