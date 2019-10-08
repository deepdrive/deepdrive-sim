
#include "Simulation/Traffic/BehaviorTree/DeepDriveBehaviorTreeHelpers.h"

#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBlackboard.h"
#include "Simulation/Traffic/Path/DeepDrivePathDefines.h"
#include "Public/Simulation/DeepDriveSimulation.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"

bool DeepDriveBehaviorTreeHelpers::isJunctionClear(DeepDriveTrafficBlackboard &blackboard)
{
	ADeepDriveSimulation *simulation = blackboard.getSimulation();

	const SDeepDriveManeuver *maneuver = blackboard.getManeuver();

	TArray<ADeepDriveAgent *> agents = simulation->getAgents(maneuver->ManeuverArea, blackboard.getAgent());

	bool isJunctionClear = true;
	if (agents.Num() > 0)
	{
		for (auto &crossTrafficRoad : maneuver->CrossTrafficRoads)
		{
			for (auto &curPath : crossTrafficRoad.Paths)
			{
				for (auto &curAgent : agents)
				{
					const FVector curAgentLoc = curAgent->GetActorLocation();
					FVector2D forward(curAgent->GetActorForwardVector());
					forward.Normalize();

					//	check if agent close to this path
					for (auto &curPathPoint : curPath)
					{
						if	(	FVector::DistSquared(curAgentLoc, curPathPoint.Location) < (10000.0f)
							&&	FVector2D::DotProduct(forward, curPathPoint.Direction) > 0.9f
							)
						{
							isJunctionClear = false;
							break;
						}
					}
					if (!isJunctionClear)
						break;
				}
				if (!isJunctionClear)
					break;
			}
			if (!isJunctionClear)
				break;
		}
	}

	return isJunctionClear;
}

float DeepDriveBehaviorTreeHelpers::calculateJunctionClearValue(DeepDriveTrafficBlackboard &blackboard)
{
	ADeepDriveSimulation *simulation = blackboard.getSimulation();

	const SDeepDriveManeuver *maneuver = blackboard.getManeuver();
	ADeepDriveAgent *agent = blackboard.getAgent();
	TArray<ADeepDriveAgent *> agents = simulation->getAgents(maneuver->ManeuverArea, agent);

	float clearValue = 1.0f;

	for(auto &curAgent : agents)
	{
		const FVector curAgentLoc = curAgent->GetActorLocation();
		FVector2D forward(curAgent->GetActorForwardVector());
		forward.Normalize();

		uint32 agentDirectionMask = 0;
		switch(curAgent->GetDirectionIndicatorState())
		{
			case EDeepDriveAgentDirectionIndicatorState::UNKNOWN:
				agentDirectionMask = 0xF;
				break;
			case EDeepDriveAgentDirectionIndicatorState::LEFT:
				agentDirectionMask = 1;
				break;
			case EDeepDriveAgentDirectionIndicatorState::OFF:
				agentDirectionMask = 2;
				break;
			case EDeepDriveAgentDirectionIndicatorState::RIGHT:
				agentDirectionMask = 4;
				break;
			case EDeepDriveAgentDirectionIndicatorState::HAZARD_LIGHTS:
				agentDirectionMask = 8;
				break;
		}

		float bestScore = 0.0f;
		for(auto &crossTraffic : maneuver->CrossTrafficRoads)
		{
			uint32 maneuverMask = 0;

			switch(crossTraffic.ManeuverType)
			{
				case EDeepDriveManeuverType::TURN_LEFT:
					maneuverMask = 1;
					break;
				case EDeepDriveManeuverType::GO_ON_STRAIGHT:
					maneuverMask = 2;
					break;
				case EDeepDriveManeuverType::TURN_RIGHT:
					maneuverMask = 4;
					break;
				case EDeepDriveManeuverType::TRAFFIC_CIRCLE:
					maneuverMask = 4;
					break;
			}

			if	(	crossTraffic.Paths.Num() > 0
				&&	(agentDirectionMask & maneuverMask) != 0
				)
			{
				const TDeepDrivePathPoints &curPath = crossTraffic.Paths[0];

				for(auto &curPathPoint : curPath)
				{
					const float maxDistance2 = 400.0f * 400.0f;
					float curDist2 = FMath::Clamp((maxDistance2 - FVector::DistSquared(curAgentLoc, curPathPoint.Location)) / maxDistance2, 0.0f, 1.0f);
					const float headingFactor = 0.5f * (FVector2D::DotProduct(forward, curPathPoint.Direction) + 1.0f);
					const float curScore = headingFactor * curDist2;
					if(curScore > bestScore)
					{
						bestScore = curScore;
					}
				}
			}
		}

		const float curClearValue = 1.0f - bestScore;
		clearValue = FMath::Min(curClearValue, clearValue);
	}

	return clearValue;
}
