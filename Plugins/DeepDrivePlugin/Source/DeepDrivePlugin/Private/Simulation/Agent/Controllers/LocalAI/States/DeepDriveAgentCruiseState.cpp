
#include "DeepDrivePluginPrivatePCH.h"
#include "Private/Simulation/Agent/Controllers/LocalAI/States/DeepDriveAgentCruiseState.h"
#include "Private/Simulation/Agent/Controllers/DeepDriveAgentSpeedController.h"
#include "Private/Simulation/Agent/Controllers/DeepDriveAgentSteeringController.h"
#include "Public/Simulation/Agent/Controllers/DeepDriveAgentLocalAIController.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"


DeepDriveAgentCruiseState::DeepDriveAgentCruiseState(DeepDriveAgentLocalAIStateMachine &stateMachine)
	: DeepDriveAgentLocalAIStateBase(stateMachine, "Cruise")
{
}


void DeepDriveAgentCruiseState::enter(DeepDriveAgentLocalAIStateMachineContext &ctx)
{
	m_WaitTimeBeforeOvertaking = ctx.wait_time_before_overtaking;
	UE_LOG(LogDeepDriveAgentLocalAIController, Log, TEXT("Agent %d Cruising"), ctx.agent.getAgentId());
}

void DeepDriveAgentCruiseState::update(DeepDriveAgentLocalAIStateMachineContext &ctx, float dT)
{
	if(m_WaitTimeBeforeOvertaking > 0.0f)
		m_WaitTimeBeforeOvertaking -= dT;

	if(isOvertakingPossible(ctx))
	{
		m_StateMachine.setNextState("PullOut");
		UE_LOG(LogDeepDriveAgentLocalAIController, Log, TEXT("Agent %d trying to overtake agent %d Pulling out"), ctx.agent.getAgentId(), ctx.agent.getNextAgent()->getAgentId() );
	}

	float desiredSpeed = ctx.local_ai_ctrl.getDesiredSpeed();
	desiredSpeed = ctx.speed_controller.limitSpeedByTrack(desiredSpeed, 1.0f);
	desiredSpeed = ctx.speed_controller.limitSpeedByNextAgent(desiredSpeed);

	ctx.speed_controller.update(dT, desiredSpeed);
	ctx.steering_controller.update(dT, desiredSpeed, 0.0f);
}

void DeepDriveAgentCruiseState::exit(DeepDriveAgentLocalAIStateMachineContext &ctx)
{
}


bool DeepDriveAgentCruiseState::isOvertakingPossible(DeepDriveAgentLocalAIStateMachineContext &ctx)
{
	bool res = false;

	if	(	ctx.configuration.MaxAgentsToOvertake > 0
		&&	m_WaitTimeBeforeOvertaking <= 0.0f
		)
	{
		float distanceToNextAgent = -1.0f;
		ADeepDriveAgent *nextAgent = ctx.agent.getNextAgent(&distanceToNextAgent);
		if (nextAgent && distanceToNextAgent <= ctx.configuration.MinPullOutDistance)
		{
			const float speedDiff = (ctx.configuration.OvertakingSpeed - nextAgent->getSpeed() * 0.036f);
			if(speedDiff > ctx.configuration.MinSpeedDifference)
			{
				float nextButOneDist = -1.0f;
				ADeepDriveAgent *nextButOne = nextAgent->getNextAgent(&nextButOneDist);
				if	(	nextButOne == &ctx.agent
					||	nextButOneDist < ctx.configuration.GapBetweenAgents
					)
				{
					// calculate pure overtaking distance
					const float pureOvertakingDistance = distanceToNextAgent + ctx.configuration.MinPullInDistance + nextAgent->getFrontBumperDistance() + ctx.agent.getBackBumperDistance();
					// calcualte time nased on speed difference
					const float overtakingDuration = pureOvertakingDistance / (speedDiff * 100.0f * 1000.0f / 3600.0f) ;
					// calculate distance covered in that time based on theoretically overtaking speed
					const float overtakingDistance = overtakingDuration * ctx.configuration.OvertakingSpeed * 100.0f * 1000.0f / 3600.0f;

					// UE_LOG(LogDeepDriveAgentLocalAIController, Log, TEXT("%f %f %f %f"), pureOvertakingDistance, speedDiff, overtakingDuration, overtakingDistance );

					res = ctx.local_ai_ctrl.isOppositeTrackClear(overtakingDistance, overtakingDuration);
				}
			}
		}
	}
	return res;
}
