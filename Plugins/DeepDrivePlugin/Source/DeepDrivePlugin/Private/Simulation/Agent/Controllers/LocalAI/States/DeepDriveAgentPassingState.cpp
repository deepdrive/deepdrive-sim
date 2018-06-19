
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
	m_AgentToPass = ctx.agent.getNextAgent(-1.0f);
	
	m_totalSpeedDifference = 0.0f;
	m_totalSpeedDifferenceCount = 0;

	m_TotalOppositeTrackClearance = 0.0f;
	m_OppositeTrackClearanceCount = 0.0f;

	UE_LOG(LogDeepDriveAgentLocalAIController, Log, TEXT(">>>> Passing agent %d/%s"), m_AgentToPass ? m_AgentToPass->getAgentId() : -1, *(m_AgentToPass ? *(m_AgentToPass->GetName()) : FString("")) );
}

void DeepDriveAgentPassingState::update(DeepDriveAgentLocalAIStateMachineContext &ctx, float dT)
{
	if	(	m_AgentToPass
		&&	ctx.agent.getNextAgent(-1.0f) != &ctx.agent
		)
	{
		const float passedDistance = ctx.local_ai_ctrl.getPassedDistance(m_AgentToPass);
		if(passedDistance > 0.0f)
			m_StateMachine.setNextState("PullIn");
		else if(isTimeToThink(dT))
		{
			if(abortOvertaking(ctx))
				m_StateMachine.setNextState("AbortOvertaking");
		}
	}
	else
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

bool DeepDriveAgentPassingState::abortOvertaking(DeepDriveAgentLocalAIStateMachineContext &ctx)
{
	bool abort = false;

	const float curSpeed = ctx.agent.getSpeedKmh();
	float speedDiff = FMath::Max(1.0f, (curSpeed - m_AgentToPass->getSpeedKmh()));
	m_totalSpeedDifference += speedDiff;
	m_totalSpeedDifferenceCount++;
	speedDiff = m_totalSpeedDifference / static_cast<float>(m_totalSpeedDifferenceCount);

	float otc = 0.0f;
	float distanceToNextAgent = -1.0f;
	ADeepDriveAgent *nextAgent = ctx.agent.getNextAgent(ctx.configuration.GapBetweenAgents, &distanceToNextAgent);
	if(nextAgent)
	{
		const float overtakingDist = distanceToNextAgent + ctx.configuration.MinPullInDistance + nextAgent->getFrontBumperDistance() + ctx.agent.getBackBumperDistance();
		//otc = ctx.local_ai_ctrl.computeOppositeTrackClearance(overtakingDist, speedDiff, curSpeed, true);
		otc = ctx.local_ai_ctrl.computeOppositeTrackClearance(curSpeed, ctx.configuration.LookAheadTime);
	}
	else
	{
		//otc = ctx.local_ai_ctrl.computeOppositeTrackClearance(ctx.configuration.GapBetweenAgents, speedDiff, curSpeed, true);
		otc = ctx.local_ai_ctrl.computeOppositeTrackClearance(curSpeed, ctx.configuration.LookAheadTime);
	}
	abort = otc < 0.10f;

	//UE_LOG(LogDeepDriveAgentLocalAIController, Log, TEXT("otc %f"), otc);


#if 0
	float distanceToNextAgent = 1.0f;
	ADeepDriveAgent *nextAgent = ctx.agent.getNextAgent(distanceToNextAgent, &distanceToNextAgent);
	if (nextAgent)
	{
		ADeepDriveAgent *nextButOne = nextAgent->getNextAgent(ctx.configuration.GapBetweenAgents);
		if(nextButOne == 0)
		{
			const float curSpeed = ctx.agent.getSpeedKmh();
			float speedDiff = FMath::Max(1.0f, (curSpeed - nextAgent->getSpeedKmh()));

			m_totalSpeedDifference += speedDiff;
			m_totalSpeedDifferenceCount++;
			speedDiff = m_totalSpeedDifference / static_cast<float>(m_totalSpeedDifferenceCount);

			float otc = ctx.local_ai_ctrl.isOppositeTrackClear(*nextAgent, distanceToNextAgent, speedDiff, curSpeed, true);
			abort = otc < 1.0f;

			m_TotalOppositeTrackClearance += otc;
			m_OppositeTrackClearanceCount += 1.0f;
			UE_LOG(LogDeepDriveAgentLocalAIController, Log, TEXT("cur %f avg %f d2n %f spdDiff %3.1f"), otc, m_TotalOppositeTrackClearance / m_OppositeTrackClearanceCount, distanceToNextAgent, speedDiff);
		}
	}
#endif
	return abort;
}
