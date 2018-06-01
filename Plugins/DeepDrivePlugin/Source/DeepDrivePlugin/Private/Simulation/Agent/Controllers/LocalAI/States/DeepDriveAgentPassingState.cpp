
#include "DeepDrivePluginPrivatePCH.h"
#include "Private/Simulation/Agent/Controllers/LocalAI/States/DeepDriveAgentPassingState.h"
#include "Private/Simulation/Agent/Controllers/DeepDriveAgentSpeedController.h"
#include "Private/Simulation/Agent/Controllers/DeepDriveAgentSteeringController.h"
#include "Public/Simulation/Agent/Controllers/DeepDriveAgentLocalAIController.h"


DeepDriveAgentPassingState::DeepDriveAgentPassingState(DeepDriveAgentLocalAIStateMachine &stateMachine)
	: DeepDriveAgentLocalAIStateBase(stateMachine, "Passing")
{

}


void DeepDriveAgentPassingState::enter(DeepDriveAgentLocalAIStateMachineContext &ctx)
{
	UE_LOG(LogDeepDriveAgentLocalAIController, Log, TEXT("Agent %d Passing"), ctx.agent.getAgentId());
}

void DeepDriveAgentPassingState::update(DeepDriveAgentLocalAIStateMachineContext &ctx, float dT)
{
	if (ctx.local_ai_ctrl.calculateAbortOvertakingScore() > 0.0f)
	{
		m_StateMachine.setNextState("AbortOvertaking");
	}
	else if (ctx.local_ai_ctrl.hasPassed(ctx.agent_to_overtake, ctx.configuration.MinPullInDistance))
	{
		m_StateMachine.setNextState("PullIn");
	}
	
	float desiredSpeed = ctx.configuration.OvertakingSpeed;
	desiredSpeed = ctx.speed_controller.limitSpeedByTrack(desiredSpeed, ctx.configuration.SpeedLimitFactor);

	ctx.speed_controller.update(dT, desiredSpeed);
	ctx.steering_controller.update(dT, desiredSpeed, ctx.side_offset);
}

void DeepDriveAgentPassingState::exit(DeepDriveAgentLocalAIStateMachineContext &ctx)
{
}

