
#include "DeepDrivePluginPrivatePCH.h"
#include "Private/Simulation/Agent/Controllers/LocalAI/States/DeepDriveAgentPullBackInState.h"
#include "Public/Simulation/Agent/Controllers/DeepDriveAgentLocalAIController.h"
#include "Private/Simulation/Agent/Controllers/DeepDriveAgentSpeedController.h"
#include "Private/Simulation/Agent/Controllers/DeepDriveAgentSteeringController.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"


DeepDriveAgentPullBackInState::DeepDriveAgentPullBackInState(DeepDriveAgentLocalAIStateMachine &stateMachine)
	: DeepDriveAgentLocalAIStateBase(stateMachine, "PullBackIn")
{

}


void DeepDriveAgentPullBackInState::enter(DeepDriveAgentLocalAIStateMachineContext &ctx)
{
	m_curOffset = ctx.side_offset;
	m_deltaOffsetFac = ctx.configuration.OvertakingOffset / ctx.configuration.ChangeLaneDuration;
	m_remainingPullInTime = m_curOffset / m_deltaOffsetFac;
	UE_LOG(LogDeepDriveAgentLocalAIController, Log, TEXT("Agent %d Pulling back In"), ctx.agent.getAgentId());
}

void DeepDriveAgentPullBackInState::update(DeepDriveAgentLocalAIStateMachineContext &ctx, float dT)
{
	m_remainingPullInTime -= dT;
	if (m_remainingPullInTime <= 0.0f)
	{
		m_StateMachine.setNextState("Cruise");
		m_curOffset = 0.0f;
	}
	else
		m_curOffset -= dT * m_deltaOffsetFac;

	float desiredSpeed = ctx.local_ai_ctrl.getDesiredSpeed();
	desiredSpeed = ctx.speed_controller.limitSpeedByTrack(desiredSpeed, ctx.configuration.SpeedLimitFactor);

	float safetyDistance = ctx.local_ai_ctrl.calculateSafetyDistance();
	float curDistanceToNext = 0.0f;
	ADeepDriveAgent *nextAgent = ctx.agent.getNextAgent(2.0f * safetyDistance, &curDistanceToNext);
	ctx.speed_controller.update(dT, desiredSpeed, nextAgent ? safetyDistance : -1.0f, curDistanceToNext);

	ctx.steering_controller.update(dT, desiredSpeed, m_curOffset);

}

void DeepDriveAgentPullBackInState::exit(DeepDriveAgentLocalAIStateMachineContext &ctx)
{
	ctx.wait_time_before_overtaking = 2.0f;
}
