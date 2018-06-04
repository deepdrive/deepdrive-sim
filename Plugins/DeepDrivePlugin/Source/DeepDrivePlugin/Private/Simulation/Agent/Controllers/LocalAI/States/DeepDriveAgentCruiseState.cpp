
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

	if	(	ctx.configuration.MaxAgentsToOvertake > 0
		&&	m_WaitTimeBeforeOvertaking <= 0.0f
		)
	{
		ADeepDriveAgent *finalAgentToOvertake = 0;
		float score = ctx.local_ai_ctrl.calculateOvertakingScore(ctx.configuration.MaxAgentsToOvertake, ctx.configuration.OvertakingSpeed, finalAgentToOvertake);
		if(finalAgentToOvertake && score > 0.0f)
		{
			ctx.agent_to_overtake = finalAgentToOvertake;
			m_StateMachine.setNextState("PullOut");
			UE_LOG(LogDeepDriveAgentLocalAIController, Log, TEXT("Agent %d trying to overtake agent %d Pulling out"), ctx.agent.getAgentId(), ctx.agent.getNextAgent()->getAgentId() );
		}
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
				const float overtakingDistance = distanceToNextAgent + nextAgent->getFrontBumperDistance() + ctx.agent.getBackBumperDistance() + ctx.configuration.MinPullInDistance;

			}
		}
	}
	return res;
}
