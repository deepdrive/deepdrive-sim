
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
}

void DeepDriveAgentPullInState::update(DeepDriveAgentLocalAIStateMachineContext &ctx, float dT)
{
	m_remainingPullInTime -= dT;
	m_curOffset -= dT * m_deltaOffsetFac;
	if (m_remainingPullInTime <= 0.0f)
	{
		m_StateMachine.setNextState("Cruise");
	}

	float desiredSpeed = ctx.configuration.OvertakingSpeed;
	desiredSpeed = ctx.speed_controller.limitSpeedByTrack(desiredSpeed, ctx.configuration.OvertakingSpeedLimitBoost);

	ctx.speed_controller.update(dT, desiredSpeed);
	ctx.steering_controller.update(dT, desiredSpeed, m_curOffset);

}

void DeepDriveAgentPullInState::exit(DeepDriveAgentLocalAIStateMachineContext &ctx)
{
}
