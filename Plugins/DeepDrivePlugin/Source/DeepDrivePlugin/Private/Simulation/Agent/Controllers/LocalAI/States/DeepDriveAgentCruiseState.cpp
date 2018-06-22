
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
	m_WaitTimeBeforeOvertaking = ctx.wait_time_before_overtaking;
	startThinkTimer(ctx.configuration.ThinkDelays.X, false);
	UE_LOG(LogDeepDriveAgentLocalAIController, Log, TEXT(">>>> Cruising Agent %d"), ctx.agent.getAgentId());
}

void DeepDriveAgentCruiseState::update(DeepDriveAgentLocalAIStateMachineContext &ctx, float dT)
{
	if(m_WaitTimeBeforeOvertaking > 0.0f)
		m_WaitTimeBeforeOvertaking -= dT;

	if(isTimeToThink(dT))
	{
		if(isOvertakingPossible(ctx))
		{
			m_StateMachine.setNextState("PullOut");
		}
	}

	float desiredSpeed = ctx.local_ai_ctrl.getDesiredSpeed();
	desiredSpeed = ctx.speed_controller.limitSpeedByTrack(desiredSpeed, 1.0f);
	float safetyDistance = ctx.local_ai_ctrl.calculateSafetyDistance();
	float curDistanceToNext = 0.0f;
	ADeepDriveAgent *nextAgent = ctx.agent.getNextAgent(2.0f * safetyDistance, &curDistanceToNext);

	if(nextAgent)
	{
		FVector2D a2a( FVector2D(nextAgent->GetActorLocation()) - FVector2D(ctx.agent.GetActorLocation()) );
		a2a.Normalize();
		FVector2D dir(ctx.agent.GetActorForwardVector());
		dir.Normalize();
		const float dot = FVector2D::DotProduct(dir, a2a);

		if(curDistanceToNext > 0.0f || dot > ctx.configuration.SafetyVsOvertakingThreshold)
			ctx.speed_controller.update(dT, desiredSpeed, safetyDistance, curDistanceToNext);
		else
			ctx.speed_controller.update(dT, desiredSpeed, -1.0f, 0.0f);
	}
	else
		ctx.speed_controller.update(dT, desiredSpeed, -1.0f, 0.0f);

	ctx.steering_controller.update(dT, desiredSpeed, 0.0f);
}

void DeepDriveAgentCruiseState::exit(DeepDriveAgentLocalAIStateMachineContext &ctx)
{
}


bool DeepDriveAgentCruiseState::isOvertakingPossible(DeepDriveAgentLocalAIStateMachineContext &ctx)
{
	bool res = false;

	if	(	ctx.configuration.MaxAgentsToOvertake > 0
		&&	m_WaitTimeBeforeOvertaking <= 0.0f
		)
	{
		float distanceToNextAgent = -1.0f;
		ADeepDriveAgent *nextAgent = ctx.agent.getNextAgent(ctx.configuration.MinPullOutDistance, &distanceToNextAgent);
		if (nextAgent)
		{
			const float agentSpdKmh = nextAgent->getSpeedKmh();
			const float maxSpeedDiff = (ctx.configuration.OvertakingSpeed - agentSpdKmh);
			if(maxSpeedDiff > ctx.configuration.MinSpeedDifference)
			{
				ADeepDriveAgent *nextButOne = nextAgent->getNextAgent(ctx.configuration.GapBetweenAgents);
				if(nextButOne == 0)
				{
					const float curSpeed = ctx.agent.getSpeedKmh();
					float otc = ctx.local_ai_ctrl.computeOppositeTrackClearance(curSpeed, ctx.configuration.LookAheadTime);
					res = otc >= 0.0f;
					if(res)
						UE_LOG(LogDeepDriveAgentLocalAIController, Log, TEXT("Overtaking %s / %d possible"), *(nextAgent->GetName()), nextAgent->getAgentId() );
				}
			}
		}
	}
	return res;
}
