
#include "DeepDrivePluginPrivatePCH.h"
#include "Private/Simulation/Agent/Controllers/LocalAI/States/DeepDriveAgentBeginOvertakingState.h"
#include "Private/Simulation/Agent/Controllers/DeepDriveAgentSplineDrivingCtrl.h"
#include "Public/Simulation/Agent/Controllers/LocalAI/DeepDriveAgentLocalAIController.h"


DeepDriveAgentBeginOvertakingState::DeepDriveAgentBeginOvertakingState(DeepDriveAgentLocalAIStateMachine &stateMachine)
	: DeepDriveAgentLocalAIStateBase(stateMachine, "BeginOvertaking")
{

}


void DeepDriveAgentBeginOvertakingState::enter(DeepDriveAgentLocalAIStateMachineContext &ctx)
{
	m_remainingPullOutTime = ctx.configuration.ChangeLaneDuration;
	m_deltaOffsetFac = ctx.configuration.OvertakingOffset  / m_remainingPullOutTime;

	ctx.overtaking_in_progess = true;
	ctx.spline_driving_ctrl.setKeepSafetyDistance(false);
}

void DeepDriveAgentBeginOvertakingState::update(DeepDriveAgentLocalAIStateMachineContext &ctx, float dT)
{
	m_remainingPullOutTime -= dT;
	m_curOffset += dT * m_deltaOffsetFac;
	if (m_remainingPullOutTime <= 0.0f)
	{
		m_StateMachine.setNextState("Overtaking");
	}
	ctx.spline_driving_ctrl.update(dT, ctx.configuration.OvertakingSpeed, m_curOffset);
}

void DeepDriveAgentBeginOvertakingState::exit(DeepDriveAgentLocalAIStateMachineContext &ctx)
{
	ctx.side_offset = m_curOffset;
}
