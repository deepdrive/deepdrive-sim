
#include "DeepDrivePluginPrivatePCH.h"
#include "Private/Simulation/Agent/Controllers/LocalAI/States/DeepDriveAgentPullOutState.h"
#include "Private/Simulation/Agent/Controllers/DeepDriveAgentSpeedController.h"
#include "Private/Simulation/Agent/Controllers/DeepDriveAgentSteeringController.h"
#include "Public/Simulation/Agent/Controllers/DeepDriveAgentLocalAIController.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"


DeepDriveAgentPullOutState::DeepDriveAgentPullOutState(DeepDriveAgentLocalAIStateMachine &stateMachine)
	: DeepDriveAgentLocalAIStateBase(stateMachine, "PullOut")
{

}


void DeepDriveAgentPullOutState::enter(DeepDriveAgentLocalAIStateMachineContext &ctx)
{
	m_PullOutTimeFactor = 1.0f / ctx.configuration.ChangeLaneDuration;
	m_PullOutAlpha = 0.0f;

	m_curOffset = 0.0f;

	startThinkTimer(ctx.configuration.ThinkDelays.Y, false);

	UE_LOG(LogDeepDriveAgentLocalAIController, Log, TEXT("Agent %d Pulling out"), ctx.agent.getAgentId());
}

void DeepDriveAgentPullOutState::update(DeepDriveAgentLocalAIStateMachineContext &ctx, float dT)
{
	m_PullOutAlpha += m_PullOutTimeFactor * dT;
	m_curOffset = FMath::Lerp(0.0f, ctx.configuration.OvertakingOffset, m_PullOutAlpha);

	float desiredSpeed = ctx.configuration.OvertakingSpeed;// ctx.local_ai_ctrl.getDesiredSpeed();

	if (m_PullOutAlpha >= 1.0f)
	{
		m_StateMachine.setNextState("Passing");
	}
	else if(isTimeToThink(dT) && abortOvertaking(ctx, ctx.agent.getSpeedKmh()))
	{
		m_StateMachine.setNextState("PullBackIn");
	}

	float safetyDistance = ctx.local_ai_ctrl.calculateSafetyDistance();
	float curDistanceToNext = 0.0f;
	ADeepDriveAgent *nextAgent = ctx.agent.getNextAgent(2.0f * safetyDistance, &curDistanceToNext);
	ctx.speed_controller.update(dT, desiredSpeed, nextAgent ? safetyDistance : -1.0f, curDistanceToNext);

	ctx.steering_controller.update(dT, desiredSpeed, m_curOffset);
}

void DeepDriveAgentPullOutState::exit(DeepDriveAgentLocalAIStateMachineContext &ctx)
{
	ctx.side_offset = m_curOffset;
}


bool DeepDriveAgentPullOutState::abortOvertaking(DeepDriveAgentLocalAIStateMachineContext &ctx, float desiredSpeed)
{
	bool abort = false;

	float distanceToNextAgent = -1.0f;
	ADeepDriveAgent *nextAgent = ctx.agent.getNextAgent(-1.0f, &distanceToNextAgent);
	if (nextAgent)
	{
		const float curSpeed = ctx.agent.getSpeedKmh();
		//const float speedDiff = (ctx.configuration.OvertakingSpeed - nextAgent->getSpeed() * 0.036f);
		const float speedDiff = FMath::Max(1.0f, (desiredSpeed - nextAgent->getSpeedKmh()));

		ADeepDriveAgent *nextButOne = nextAgent->getNextAgent(ctx.configuration.GapBetweenAgents);
		if(nextButOne == 0)
		{
			float otc = ctx.local_ai_ctrl.isOppositeTrackClear(*nextAgent, distanceToNextAgent, speedDiff, curSpeed, true);
			abort = otc < 1.0f;
		}
		else
			abort = nextButOne == &ctx.agent;
	}

	return abort;
}
