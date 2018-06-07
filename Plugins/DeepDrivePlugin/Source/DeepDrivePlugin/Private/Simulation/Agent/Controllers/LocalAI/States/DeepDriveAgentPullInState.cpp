
#include "DeepDrivePluginPrivatePCH.h"
#include "Private/Simulation/Agent/Controllers/LocalAI/States/DeepDriveAgentPullInState.h"
#include "Public/Simulation/Agent/Controllers/DeepDriveAgentLocalAIController.h"
#include "Private/Simulation/Agent/Controllers/DeepDriveAgentSpeedController.h"
#include "Private/Simulation/Agent/Controllers/DeepDriveAgentSteeringController.h"


DeepDriveAgentPullInState::DeepDriveAgentPullInState(DeepDriveAgentLocalAIStateMachine &stateMachine)
	: DeepDriveAgentLocalAIStateBase(stateMachine, "PullIn")
{

}


void DeepDriveAgentPullInState::enter(DeepDriveAgentLocalAIStateMachineContext &ctx)
{
	m_remainingPullInTime = ctx.configuration.ChangeLaneDuration;
	m_curOffset = ctx.side_offset;
	m_deltaOffsetFac = ctx.configuration.OvertakingOffset / m_remainingPullInTime;
	UE_LOG(LogDeepDriveAgentLocalAIController, Log, TEXT("Agent %d Pulling In"), ctx.agent.getAgentId());
}

void DeepDriveAgentPullInState::update(DeepDriveAgentLocalAIStateMachineContext &ctx, float dT)
{
	m_remainingPullInTime -= dT;
	m_curOffset -= dT * m_deltaOffsetFac;
	if (m_remainingPullInTime <= 0.0f)
	{
		m_StateMachine.setNextState("Cruise");
	}

	float desiredSpeed = ctx.local_ai_ctrl.getDesiredSpeed();
	desiredSpeed = ctx.speed_controller.limitSpeedByTrack(desiredSpeed, ctx.configuration.SpeedLimitFactor);
	desiredSpeed = ctx.speed_controller.limitSpeedByNextAgent(desiredSpeed);

	ctx.speed_controller.update(dT, desiredSpeed);
	ctx.steering_controller.update(dT, desiredSpeed, m_curOffset);

	ADeepDriveAgent *next = ctx.agent.getNextAgent();

	UE_LOG(LogDeepDriveAgentLocalAIController, Log, TEXT("Agent %d Pulling In spd %f %s"), ctx.agent.getAgentId(), desiredSpeed, *(next->GetName()) );

}

void DeepDriveAgentPullInState::exit(DeepDriveAgentLocalAIStateMachineContext &ctx)
{
	ctx.wait_time_before_overtaking = 2.0f;
}
