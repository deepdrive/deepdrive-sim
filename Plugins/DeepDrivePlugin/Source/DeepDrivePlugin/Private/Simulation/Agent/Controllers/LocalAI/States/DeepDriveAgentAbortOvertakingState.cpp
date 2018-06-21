
#include "DeepDrivePluginPrivatePCH.h"
#include "Private/Simulation/Agent/Controllers/LocalAI/States/DeepDriveAgentAbortOvertakingState.h"
#include "Public/Simulation/Agent/Controllers/DeepDriveAgentLocalAIController.h"
#include "Private/Simulation/Agent/Controllers/DeepDriveAgentSpeedController.h"
#include "Private/Simulation/Agent/Controllers/DeepDriveAgentSteeringController.h"
#include "Public/Simulation/Agent/Controllers/DeepDriveAgentLocalAIController.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"


DeepDriveAgentAbortOvertakingState::DeepDriveAgentAbortOvertakingState(DeepDriveAgentLocalAIStateMachine &stateMachine)
	: DeepDriveAgentLocalAIStateBase(stateMachine, "AbortOvertaking")
{

}


void DeepDriveAgentAbortOvertakingState::enter(DeepDriveAgentLocalAIStateMachineContext &ctx)
{
	float distanceToNextAgent = -1.0f;
	float distanceToPrevAgent = -1.0f;
	ADeepDriveAgent *nextAgent = ctx.agent.getNextAgent(ctx.configuration.GapBetweenAgents, &distanceToNextAgent);
	ADeepDriveAgent *prevAgent = ctx.agent.getPrevAgent(ctx.configuration.GapBetweenAgents, &distanceToPrevAgent);

	if(nextAgent == 0 && prevAgent == 0)
	{
		m_SubState = PullIn;
	}
	else
	{
		m_SubState = FallBack;
	}

	m_PullInTimeFactor = 1.0f / ctx.configuration.ChangeLaneDuration;
	m_PullInAlpha = 1.0f;

	UE_LOG(LogDeepDriveAgentLocalAIController, Log, TEXT("AbortOvertaking next %s %f prev %s %f"), *(nextAgent ? nextAgent->GetName() : FString("XXX")), distanceToNextAgent, *(prevAgent ? prevAgent->GetName() : FString("XXX")), distanceToPrevAgent );
}

void DeepDriveAgentAbortOvertakingState::update(DeepDriveAgentLocalAIStateMachineContext &ctx, float dT)
{
	switch (m_SubState)
	{
		case FallBack:
			fallBack(ctx, dT);
			#if 0
			{
				ADeepDriveAgent *refAgent = getReferenceAgent(ctx.agent, ctx.configuration.GapBetweenAgents, ctx.configuration.GapBetweenAgents);
				//UE_LOG(LogDeepDriveAgentLocalAIController, Log, TEXT("Falling back %f"), distanceToNextAgent);

				if(refAgent)
				{
					float brake = 1.0f;
					float desiredSpeed = 5.0f;
					desiredSpeed = 0.75f * refAgent->getSpeedKmh();
					float curSpeed = ctx.agent.getSpeedKmh();
					if(curSpeed > desiredSpeed)
						brake = FMath::SmoothStep(1.0f, 1.25f, curSpeed / desiredSpeed);
					else
						brake = 0.0f;
					if(brake > 0.0f)
						ctx.speed_controller.brake(brake);
					else
						ctx.speed_controller.update(dT, desiredSpeed, -1.0f, 0.0f);

					UE_LOG(LogDeepDriveAgentLocalAIController, Log, TEXT("Falling back Ref agent %s desiredSpeed %3.1f curSpeed %3.1f brake %f"), *(refAgent->GetName()), desiredSpeed, curSpeed, brake);


					ctx.steering_controller.update(dT, desiredSpeed, ctx.side_offset);
				}
				else
					m_SubState = PullIn;				
			}
#endif
			break;

		case PullIn:
			pullIn(ctx, dT);
			break;
	}

}

void DeepDriveAgentAbortOvertakingState::fallBack(DeepDriveAgentLocalAIStateMachineContext &ctx, float dT)
{
	float distance = -1.0f;
	ADeepDriveAgent *refAgent = getReferenceAgent(ctx.agent, ctx.configuration.GapBetweenAgents, ctx.configuration.GapBetweenAgents, distance);

	if(refAgent == 0)
		m_SubState = PullIn;
	else
	{
		FVector2D dir2Ref = FVector2D(refAgent->GetActorLocation() - ctx.agent.GetActorLocation());
		dir2Ref.Normalize();
		FVector2D forward(ctx.agent.GetActorForwardVector());
		forward.Normalize();

		const float dot = FVector2D::DotProduct(forward, dir2Ref);

		if	(	(dot > 0.0f && distance > 0.0f)
			||	dot > 0.7f
			)
		{
			//UE_LOG(LogDeepDriveAgentLocalAIController, Log, TEXT("Start pulling in: Ref agent %s Dot %f Dist %f"), *(refAgent->GetName()), dot, distance);
			m_SubState = PullIn;
		}
		else
		{
			float desiredSpeed = 0.7f * refAgent->getSpeedKmh();
			float curSpeed = ctx.agent.getSpeedKmh();
			const float ratio = curSpeed / desiredSpeed;
			//float brake = ratio > 1.1f ? (1.0f - FMath::SmoothStep(0.0f, 0.8f, dot)) : 0.0f;
			float brake = ratio > 1.1f ? 1.0f : 0.0f;

			if(brake > 0.0f)
			{
				ctx.speed_controller.brake(brake);
				//ctx.speed_controller.update(dT, 0.75f * desiredSpeed, -1.0f, 0.0f);
			}
			else
				ctx.speed_controller.update(dT, desiredSpeed, -1.0f, 0.0f);

			UE_LOG(LogDeepDriveAgentLocalAIController, Log, TEXT("Falling back: Ref agent %s Dot %f Dist %f desiredSpeed %3.1f curSpeed %3.1f brake %f"), *(refAgent->GetName()), dot, distance, desiredSpeed, curSpeed, brake);
			ctx.steering_controller.update(dT, desiredSpeed, ctx.side_offset);
		}

	}
}

void DeepDriveAgentAbortOvertakingState::pullIn(DeepDriveAgentLocalAIStateMachineContext &ctx, float dT)
{
	m_PullInAlpha -= m_PullInTimeFactor * dT;
	m_curOffset = FMath::Lerp(0.0f, ctx.side_offset, m_PullInAlpha);

	if (m_PullInAlpha <= 0.0f)
	{
		m_StateMachine.setNextState("Cruise");
	}

	float desiredSpeed = ctx.local_ai_ctrl.getDesiredSpeed();
	desiredSpeed = ctx.speed_controller.limitSpeedByTrack(desiredSpeed, ctx.configuration.SpeedLimitFactor);

	float safetyDistance = ctx.local_ai_ctrl.calculateSafetyDistance();
	float curDistanceToNext = 0.0f;
	ADeepDriveAgent *nextAgent = ctx.agent.getNextAgent(2.0f * safetyDistance, &curDistanceToNext);
	ctx.speed_controller.update(dT, desiredSpeed, nextAgent ? safetyDistance : -1.0f, curDistanceToNext);
	ctx.steering_controller.update(dT, desiredSpeed, m_curOffset);
}

void DeepDriveAgentAbortOvertakingState::exit(DeepDriveAgentLocalAIStateMachineContext &ctx)
{
	ctx.wait_time_before_overtaking = 4.0f;
	ctx.side_offset = m_curOffset;
}


ADeepDriveAgent* DeepDriveAgentAbortOvertakingState::getReferenceAgent(ADeepDriveAgent &self, float forwardDist, float backwardDist, float &distance)
{
	ADeepDriveAgent *refAgent = 0;

	float distanceToNextAgent = -1.0f;
	float distanceToPrevAgent = -1.0f;
	ADeepDriveAgent *nextAgent = self.getNextAgent(forwardDist, &distanceToNextAgent);
	ADeepDriveAgent *prevAgent = self.getPrevAgent(backwardDist, &distanceToPrevAgent);

	if(prevAgent)
	{
		refAgent = prevAgent;
		distance = distanceToPrevAgent;
	}
	else if(nextAgent)
	{
		refAgent = nextAgent;
		distance = distanceToNextAgent;
	}

	return refAgent;
}
