
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
	startThinkTimer(ctx.configuration.ThinkDelays.X, false);
	UE_LOG(LogDeepDriveAgentLocalAIController, Log, TEXT("Agent %d Cruising"), ctx.agent.getAgentId());
}

void DeepDriveAgentCruiseState::update(DeepDriveAgentLocalAIStateMachineContext &ctx, float dT)
{
	if(m_WaitTimeBeforeOvertaking > 0.0f)
		m_WaitTimeBeforeOvertaking -= dT;

	if(isTimeToThink(dT))
	{
		if(isOvertakingPossible(ctx))
		{
			m_StateMachine.setNextState("PullOut");
			UE_LOG(LogDeepDriveAgentLocalAIController, Log, TEXT("Agent %d trying to overtake agent %d Pulling out"), ctx.agent.getAgentId(), ctx.agent.getNextAgent()->getAgentId() );
		}
	}

	float desiredSpeed = ctx.local_ai_ctrl.getDesiredSpeed();
	desiredSpeed = ctx.speed_controller.limitSpeedByTrack(desiredSpeed, 1.0f);

	//desiredSpeed = ctx.speed_controller.limitSpeedByNextAgent(desiredSpeed);

	float curDistanceToNext = 0.0f;
	float safetyDistance = ctx.local_ai_ctrl.calculateSafetyDistance(&curDistanceToNext);

	//UE_LOG(LogDeepDriveAgentLocalAIController, Log, TEXT("Agent %s sfty %f cur %f"), *(ctx.agent.GetName()), safetyDistance, curDistanceToNext );

	ctx.speed_controller.update(dT, desiredSpeed, safetyDistance, curDistanceToNext);
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
			const float agentSpdKmh = nextAgent->getSpeedKmh();
			const float maxSpeedDiff = (ctx.configuration.OvertakingSpeed - agentSpdKmh);
			if(maxSpeedDiff > ctx.configuration.MinSpeedDifference)
			{
				float nextButOneDist = -1.0f;
				ADeepDriveAgent *nextButOne = nextAgent->getNextAgent(&nextButOneDist);
				if	(	nextButOne == &ctx.agent
					||	nextButOneDist > ctx.configuration.GapBetweenAgents
					)
				{
					const float curSpeed = ctx.agent.getSpeedKmh();
					const float overtakingSpeed = 0.5f * (curSpeed + ctx.configuration.OvertakingSpeed);
					float otc = ctx.local_ai_ctrl.isOppositeTrackClear(*nextAgent, distanceToNextAgent, overtakingSpeed - agentSpdKmh, curSpeed, true);
					res = otc >= 1.0f;
				}
			}
		}
	}
	return res;
}
