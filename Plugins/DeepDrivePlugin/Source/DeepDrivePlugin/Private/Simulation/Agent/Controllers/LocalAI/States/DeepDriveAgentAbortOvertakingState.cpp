
#include "DeepDrivePluginPrivatePCH.h"
#include "Private/Simulation/Agent/Controllers/LocalAI/States/DeepDriveAgentAbortOvertakingState.h"
#include "Private/Simulation/Agent/Controllers/DeepDriveAgentSplineDrivingCtrl.h"
#include "Public/Simulation/Agent/Controllers/LocalAI/DeepDriveAgentLocalAIController.h"


DeepDriveAgentAbortOvertakingState::DeepDriveAgentAbortOvertakingState(DeepDriveAgentLocalAIStateMachine &stateMachine)
	: DeepDriveAgentLocalAIStateBase(stateMachine, "AbortOvertaking")
{

}


void DeepDriveAgentAbortOvertakingState::enter(DeepDriveAgentLocalAIStateMachineContext &ctx)
{
}

void DeepDriveAgentAbortOvertakingState::update(DeepDriveAgentLocalAIStateMachineContext &ctx, float dT)
{
	ctx.spline_driving_ctrl.update(dT, ctx.local_ai_ctrl.DesiredSpeed, 0.0f);

}

void DeepDriveAgentAbortOvertakingState::exit(DeepDriveAgentLocalAIStateMachineContext &ctx)
{
}
