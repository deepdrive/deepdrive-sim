
#include "DeepDrivePluginPrivatePCH.h"
#include "Private/Simulation/Agent/Controllers/LocalAI/States/DeepDriveAgentFinishOvertakingState.h"
#include "Private/Simulation/Agent/Controllers/DeepDriveAgentSplineDrivingCtrl.h"
#include "Public/Simulation/Agent/Controllers/LocalAI/DeepDriveAgentLocalAIController.h"


DeepDriveAgentFinishOvertakingState::DeepDriveAgentFinishOvertakingState(DeepDriveAgentLocalAIStateMachine &stateMachine)
	: DeepDriveAgentLocalAIStateBase(stateMachine, "FinishOvertaking")
{

}


void DeepDriveAgentFinishOvertakingState::enter(DeepDriveAgentLocalAIStateMachineContext &ctx)
{
	m_remainingPullInTime = ctx.configuration.ChangeLaneDuration;
	m_curOffset = ctx.side_offset;
	m_deltaOffsetFac = ctx.configuration.OvertakingOffset / m_remainingPullInTime;
}

void DeepDriveAgentFinishOvertakingState::update(DeepDriveAgentLocalAIStateMachineContext &ctx, float dT)
{
	m_remainingPullInTime -= dT;
	m_curOffset -= dT * m_deltaOffsetFac;
	if (m_remainingPullInTime <= 0.0f)
	{
		m_StateMachine.setNextState("Cruising");
	}
	ctx.spline_driving_ctrl.update(dT, ctx.configuration.OvertakingSpeed, m_curOffset);

}

void DeepDriveAgentFinishOvertakingState::exit(DeepDriveAgentLocalAIStateMachineContext &ctx)
{
}
