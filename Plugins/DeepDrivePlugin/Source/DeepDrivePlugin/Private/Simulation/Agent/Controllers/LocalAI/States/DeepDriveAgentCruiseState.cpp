
#include "DeepDrivePluginPrivatePCH.h"
#include "Private/Simulation/Agent/Controllers/LocalAI/States/DeepDriveAgentCruiseState.h"
#include "Private/Simulation/Agent/Controllers/DeepDriveAgentSpeedController.h"
#include "Private/Simulation/Agent/Controllers/DeepDriveAgentSteeringController.h"
#include "Public/Simulation/Agent/Controllers/DeepDriveAgentLocalAIController.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"


DeepDriveAgentCruiseState::DeepDriveAgentCruiseState(DeepDriveAgentLocalAIStateMachine &stateMachine)
	: DeepDriveAgentLocalAIStateBase(stateMachine, "Cruise")
{
}


void DeepDriveAgentCruiseState::enter(DeepDriveAgentLocalAIStateMachineContext &ctx)
{
}

void DeepDriveAgentCruiseState::update(DeepDriveAgentLocalAIStateMachineContext &ctx, float dT)
{
	if	(	ctx.configuration.OvertakingEnabled
		&&	ctx.local_ai_ctrl.calculateOvertakingScore() > 0.0f
		)
	{
		m_StateMachine.setNextState("PullOut");
		UE_LOG(LogDeepDriveAgentLocalAIController, Log, TEXT("Agent %d trying to overtake agent %d Pulling out"), ctx.agent.getAgentId(), ctx.agent.getNextAgent()->getAgentId() );
	}

	float desiredSpeed = ctx.local_ai_ctrl.getDesiredSpeed();
	desiredSpeed = ctx.speed_controller.limitSpeedByTrack(desiredSpeed, 1.0f);
	desiredSpeed = ctx.speed_controller.limitSpeedByNextAgent(desiredSpeed);

	ctx.speed_controller.update(dT, desiredSpeed);
	ctx.steering_controller.update(dT, desiredSpeed, 0.0f);
}

void DeepDriveAgentCruiseState::exit(DeepDriveAgentLocalAIStateMachineContext &ctx)
{
}
