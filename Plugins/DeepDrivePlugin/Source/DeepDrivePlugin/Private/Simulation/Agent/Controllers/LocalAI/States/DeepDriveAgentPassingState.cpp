
#include "DeepDrivePluginPrivatePCH.h"
#include "Private/Simulation/Agent/Controllers/LocalAI/States/DeepDriveAgentPassingState.h"
#include "Private/Simulation/Agent/Controllers/DeepDriveAgentSpeedController.h"
#include "Private/Simulation/Agent/Controllers/DeepDriveAgentSteeringController.h"
#include "Public/Simulation/Agent/Controllers/DeepDriveAgentLocalAIController.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"


DeepDriveAgentPassingState::DeepDriveAgentPassingState(DeepDriveAgentLocalAIStateMachine &stateMachine)
	: DeepDriveAgentLocalAIStateBase(stateMachine, "Passing")
{

}


void DeepDriveAgentPassingState::enter(DeepDriveAgentLocalAIStateMachineContext &ctx)
{
	startThinkTimer(ctx.configuration.ThinkDelays.Z, false);
	m_AgentToPass = ctx.agent.getNextAgent();
	m_TotalOppositeTrackClearance = 0.0f;
	m_OppositeTrackClearanceCount = 0.0f;


	UE_LOG(LogDeepDriveAgentLocalAIController, Log, TEXT("Agent %d Passing"), m_AgentToPass ? m_AgentToPass->getAgentId() : -1);
}

void DeepDriveAgentPassingState::update(DeepDriveAgentLocalAIStateMachineContext &ctx, float dT)
{
	if(m_AgentToPass == 0)
			m_StateMachine.setNextState("PullIn");
	else if(isTimeToThink(dT))
	{
		if (abortOvertaking(ctx))
		{
			m_StateMachine.setNextState("AbortOvertaking");
		}
		else if (ctx.local_ai_ctrl.hasPassed(m_AgentToPass, ctx.configuration.MinPullInDistance))
		{
			m_StateMachine.setNextState("PullIn");
		}
	}
		
	float desiredSpeed = ctx.configuration.OvertakingSpeed;
	desiredSpeed = ctx.speed_controller.limitSpeedByTrack(desiredSpeed, ctx.configuration.SpeedLimitFactor);

	ctx.speed_controller.update(dT, desiredSpeed);
	ctx.steering_controller.update(dT, desiredSpeed, ctx.side_offset);
}

void DeepDriveAgentPassingState::exit(DeepDriveAgentLocalAIStateMachineContext &ctx)
{
}

bool DeepDriveAgentPassingState::abortOvertaking(DeepDriveAgentLocalAIStateMachineContext &ctx)
{
	bool abort = false;

	float distanceToNextAgent = -1.0f;
	ADeepDriveAgent *nextAgent = ctx.agent.getNextAgent(&distanceToNextAgent);
	if (nextAgent)
	{
		const float curSpeed = ctx.agent.getSpeedKmh();
		const float speedDiff = FMath::Max(1.0f, (curSpeed - nextAgent->getSpeedKmh()));
		float nextButOneDist = -1.0f;
		ADeepDriveAgent *nextButOne = nextAgent->getNextAgent(&nextButOneDist);
		if	(	nextButOne != &ctx.agent
			||	nextButOneDist > ctx.configuration.GapBetweenAgents
			)
		{
			float otc = ctx.local_ai_ctrl.isOppositeTrackClear(*nextAgent, distanceToNextAgent, speedDiff, curSpeed, true);
			abort = otc < 1.0f;

			m_TotalOppositeTrackClearance += otc;
			m_OppositeTrackClearanceCount += 1.0f;
			UE_LOG(LogDeepDriveAgentLocalAIController, Log, TEXT("cur %f avg %f"), otc, m_TotalOppositeTrackClearance / m_OppositeTrackClearanceCount);
		}
		else
			abort = nextButOne == &ctx.agent;
	}

	return abort;
}
