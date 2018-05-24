
#include "DeepDrivePluginPrivatePCH.h"
#include "Private/Simulation/Agent/Controllers/LocalAI/States/DeepDriveAgentCruisingState.h"
#include "Private/Simulation/Agent/Controllers/DeepDriveAgentSplineDrivingCtrl.h"
#include "Public/Simulation/Agent/Controllers/LocalAI/DeepDriveAgentLocalAIController.h"


DeepDriveAgentCruisingState::DeepDriveAgentCruisingState(DeepDriveAgentLocalAIStateMachine &stateMachine)
	: DeepDriveAgentLocalAIStateBase(stateMachine, "Cruising")
{

}


void DeepDriveAgentCruisingState::enter(DeepDriveAgentLocalAIStateMachineContext &ctx)
{
}

void DeepDriveAgentCruisingState::update(DeepDriveAgentLocalAIStateMachineContext &ctx, float dT)
{
	if(ctx.overtaking_score > 0.0f)
	{
		m_StateMachine.setNextState("BeginOvertaking");
	}
	ctx.spline_driving_ctrl.update(dT, ctx.local_ai_ctrl.DesiredSpeed, 0.0f);

}

void DeepDriveAgentCruisingState::exit(DeepDriveAgentLocalAIStateMachineContext &ctx)
{
}
