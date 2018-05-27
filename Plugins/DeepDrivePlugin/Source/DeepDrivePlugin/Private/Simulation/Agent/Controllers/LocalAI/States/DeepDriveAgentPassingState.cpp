
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
}

void DeepDriveAgentPassingState::update(DeepDriveAgentLocalAIStateMachineContext &ctx, float dT)
{
	if (ctx.local_ai_ctrl.calculateAbortOvertakingScore() > 0.0f)
	{
		m_StateMachine.setNextState("AbortOvertaking");
	}
	else if (hasPassed())
	{
		m_StateMachine.setNextState("PullIn");
	}
	
	float desiredSpeed = ctx.configuration.OvertakingSpeed;
	desiredSpeed = ctx.speed_controller.limitSpeedByTrack(desiredSpeed, ctx.configuration.OvertakingSpeedLimitBoost);

	ctx.speed_controller.update(dT, desiredSpeed);
	ctx.steering_controller.update(dT, desiredSpeed, ctx.side_offset);
}

void DeepDriveAgentPassingState::exit(DeepDriveAgentLocalAIStateMachineContext &ctx)
{
}

bool DeepDriveAgentPassingState::hasPassed()
{
	return false;
}