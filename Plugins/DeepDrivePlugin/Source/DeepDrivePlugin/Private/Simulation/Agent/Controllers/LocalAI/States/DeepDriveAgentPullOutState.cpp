
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
	m_remainingPullOutTime = ctx.configuration.ChangeLaneDuration;
	m_deltaOffsetFac = ctx.configuration.OvertakingOffset  / m_remainingPullOutTime;
	m_curOffset = 0.0f;

	UE_LOG(LogDeepDriveAgentLocalAIController, Log, TEXT("Agent %d Pulling out"), ctx.agent.getAgentId());
}

void DeepDriveAgentPullOutState::update(DeepDriveAgentLocalAIStateMachineContext &ctx, float dT)
{
	m_remainingPullOutTime -= dT;
	m_curOffset += dT * m_deltaOffsetFac;

	if(abortOvertaking(ctx))
	{
		m_StateMachine.setNextState("PullBackIn");
	}
	else if (m_remainingPullOutTime <= 0.0f)
	{
		m_StateMachine.setNextState("Passing");
	}

	float desiredSpeed = ctx.local_ai_ctrl.getDesiredSpeed();
	desiredSpeed = ctx.speed_controller.limitSpeedByTrack(desiredSpeed, ctx.configuration.SpeedLimitFactor);

	ctx.speed_controller.update(dT, desiredSpeed);
	ctx.steering_controller.update(dT, desiredSpeed, m_curOffset);
}

void DeepDriveAgentPullOutState::exit(DeepDriveAgentLocalAIStateMachineContext &ctx)
{
	ctx.side_offset = m_curOffset;
}


bool DeepDriveAgentPullOutState::abortOvertaking(DeepDriveAgentLocalAIStateMachineContext &ctx)
{
	bool abort = false;

	float distanceToNextAgent = -1.0f;
	ADeepDriveAgent *nextAgent = ctx.agent.getNextAgent(&distanceToNextAgent);
	if (nextAgent)
	{
		const float curSpeed = ctx.agent.getSpeed() * 0.036f;
		//const float speedDiff = (ctx.configuration.OvertakingSpeed - nextAgent->getSpeed() * 0.036f);
		const float speedDiff = FMath::Max(1.0f, (curSpeed - nextAgent->getSpeed() * 0.036f));
		float nextButOneDist = -1.0f;
		ADeepDriveAgent *nextButOne = nextAgent->getNextAgent(&nextButOneDist);
		if	(	nextButOne != &ctx.agent
			||	nextButOneDist > ctx.configuration.GapBetweenAgents
			)
		{
			float otc = ctx.local_ai_ctrl.isOppositeTrackClear(*nextAgent, distanceToNextAgent, speedDiff, curSpeed, true);
			abort = otc < 1.0f;
		}
		else
			abort = nextButOne == &ctx.agent;
	}

	return abort;
}
