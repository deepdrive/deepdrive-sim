
#include "DeepDrivePluginPrivatePCH.h"
#include "Private/Simulation/Agent/Controllers/LocalAI/States/DeepDriveAgentPullInState.h"
#include "Public/Simulation/Agent/Controllers/DeepDriveAgentLocalAIController.h"
#include "Private/Simulation/Agent/Controllers/DeepDriveAgentSpeedController.h"
#include "Private/Simulation/Agent/Controllers/DeepDriveAgentSteeringController.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"


DeepDriveAgentPullInState::DeepDriveAgentPullInState(DeepDriveAgentLocalAIStateMachine &stateMachine)
	: DeepDriveAgentLocalAIStateBase(stateMachine, "PullIn")
{

}


void DeepDriveAgentPullInState::enter(DeepDriveAgentLocalAIStateMachineContext &ctx)
{
	m_PullInTimeFactor = 1.0f / ctx.configuration.ChangeLaneDuration;
	m_PullInAlpha = 1.0f;

	UE_LOG(LogDeepDriveAgentLocalAIController, Log, TEXT(">>>> Pulling In Agent %d"), ctx.agent.GetAgentId());
}

void DeepDriveAgentPullInState::update(DeepDriveAgentLocalAIStateMachineContext &ctx, float dT)
{
	m_PullInAlpha -= m_PullInTimeFactor * dT;
	m_curOffset = FMath::Lerp(0.0f, ctx.side_offset, m_PullInAlpha);

	if (m_PullInAlpha <= 0.0f)
	{
		m_StateMachine.setNextState("Cruise");
	}

	float desiredSpeed = ctx.local_ai_ctrl.getDesiredSpeed();
	desiredSpeed = ctx.speed_controller.limitSpeedByTrack(desiredSpeed, ctx.configuration.SpeedLimitFactor);

	float safetyDistance = ctx.local_ai_ctrl.calculateSafetyDistance();
	float curDistanceToNext = 0.0f;
	ADeepDriveAgent *nextAgent = ctx.agent.getNextAgent(2.0f * safetyDistance, &curDistanceToNext);
	ctx.speed_controller.update(dT, desiredSpeed, nextAgent ? safetyDistance : -1.0f, curDistanceToNext);

	ctx.steering_controller.update(dT, desiredSpeed, m_curOffset);
}

void DeepDriveAgentPullInState::exit(DeepDriveAgentLocalAIStateMachineContext &ctx)
{
	ctx.wait_time_before_overtaking = 2.0f;
	ctx.side_offset = m_curOffset;
	ctx.local_ai_ctrl.setIsPassing(false);
}
