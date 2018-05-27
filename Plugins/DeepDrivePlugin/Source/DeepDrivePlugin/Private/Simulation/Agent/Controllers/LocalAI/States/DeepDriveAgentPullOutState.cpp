
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

	ctx.overtaking_in_progess = true;
}

void DeepDriveAgentPullOutState::update(DeepDriveAgentLocalAIStateMachineContext &ctx, float dT)
{
	m_remainingPullOutTime -= dT;
	m_curOffset += dT * m_deltaOffsetFac;

	//UE_LOG(LogDeepDriveAgentLocalAIController, Log, TEXT("Pulling out: Agent %d time %f dist %f"), ctx.agent.getAgentId(), m_remainingPullOutTime, m_curOffset);

	if (m_remainingPullOutTime <= 0.0f)
	{
		m_StateMachine.setNextState("Passing");
	}

	float desiredSpeed = ctx.local_ai_ctrl.getDesiredSpeed();
	desiredSpeed = ctx.speed_controller.limitSpeedByTrack(desiredSpeed, ctx.configuration.OvertakingSpeedLimitBoost);

	ctx.speed_controller.update(dT, desiredSpeed);
	ctx.steering_controller.update(dT, desiredSpeed, m_curOffset);
}

void DeepDriveAgentPullOutState::exit(DeepDriveAgentLocalAIStateMachineContext &ctx)
{
	ctx.side_offset = m_curOffset;
}
