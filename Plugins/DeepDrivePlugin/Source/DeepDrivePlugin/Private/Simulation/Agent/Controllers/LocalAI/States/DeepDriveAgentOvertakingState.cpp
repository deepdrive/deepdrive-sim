
#include "DeepDrivePluginPrivatePCH.h"
#include "Private/Simulation/Agent/Controllers/LocalAI/States/DeepDriveAgentOvertakingState.h"
#include "Private/Simulation/Agent/Controllers/DeepDriveAgentSplineDrivingCtrl.h"
#include "Public/Simulation/Agent/Controllers/LocalAI/DeepDriveAgentLocalAIController.h"


DeepDriveAgentOvertakingState::DeepDriveAgentOvertakingState(DeepDriveAgentLocalAIStateMachine &stateMachine)
	: DeepDriveAgentLocalAIStateBase(stateMachine, "Overtaking")
{

}


void DeepDriveAgentOvertakingState::enter(DeepDriveAgentLocalAIStateMachineContext &ctx)
{
	m_remainingOvertakingTime = ctx.configuration.OvertakingDuration;
}

void DeepDriveAgentOvertakingState::update(DeepDriveAgentLocalAIStateMachineContext &ctx, float dT)
{
	m_remainingOvertakingTime -= dT;
	if (m_remainingOvertakingTime <= 0.0f)
	{
		m_StateMachine.setNextState("FinishOvertaking");
	}
	ctx.spline_driving_ctrl.update(dT, ctx.configuration.OvertakingSpeed, ctx.side_offset);
}

void DeepDriveAgentOvertakingState::exit(DeepDriveAgentLocalAIStateMachineContext &ctx)
{
}
