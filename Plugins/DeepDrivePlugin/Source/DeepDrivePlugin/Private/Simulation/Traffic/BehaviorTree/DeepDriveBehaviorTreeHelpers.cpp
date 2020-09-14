
#include "Simulation/Traffic/BehaviorTree/DeepDriveBehaviorTreeHelpers.h"

#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBlackboard.h"
#include "Simulation/Traffic/Path/DeepDrivePathDefines.h"
#include "Public/Simulation/DeepDriveSimulation.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"
#include "Simulation/TrafficLight/DeepDriveTrafficLight.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoadNetworkComponent.h"

float DeepDriveBehaviorTreeHelpers::calculateJunctionClearValue_new(DeepDriveTrafficBlackboard &blackboard, bool ignoreTrafficLights)
{
	float clearValue = 1.0f;

	ADeepDriveSimulation *simulation = blackboard.getSimulation();
	const SDeepDriveManeuver *maneuver = blackboard.getManeuver();
	ADeepDriveAgent *egoAgent = blackboard.getAgent();
	const SDeepDriveJunction &junction = simulation->RoadNetwork->getRoadNetwork().Junctions[maneuver->JunctionId];
	const float estimatedJunctionClearTime = 4.0f;
	TDeepDrivePredictedPath egoPredictedPath;
	egoAgent->getPredictedPath(estimatedJunctionClearTime, egoPredictedPath);

	for (auto &crossTraffic : maneuver->CrossTrafficRoads)
	{
		TArray<ADeepDriveAgent*> agents;
		junction.getRelevantAgents(crossTraffic.FromLinkId, crossTraffic.ToLinkId, egoAgent, agents);

		for(auto &curAgent : agents)
		{
			if	(	crossTraffic.TrafficLight == 0
				||	crossTraffic.TrafficLight->CurrentPhase != EDeepDriveTrafficLightPhase::RED
				)
			{
				TDeepDrivePredictedPath curPredictedPath;
				curAgent->getPredictedPath(estimatedJunctionClearTime, egoPredictedPath);

			}
		}
	}

	return clearValue;
}

float DeepDriveBehaviorTreeHelpers::calculateJunctionClearValue(DeepDriveTrafficBlackboard &blackboard, bool ignoreTrafficLights)
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
		switch(curAgent->GetTurnSignalState())
		{
			case EDeepDriveAgentTurnSignalState::UNKNOWN:
				agentDirectionMask = 0xF;
				break;
			case EDeepDriveAgentTurnSignalState::LEFT:
				agentDirectionMask = 1;
				break;
			case EDeepDriveAgentTurnSignalState::OFF:
				agentDirectionMask = 2;
				break;
			case EDeepDriveAgentTurnSignalState::RIGHT:
				agentDirectionMask = 4;
				break;
			case EDeepDriveAgentTurnSignalState::HAZARD_LIGHTS:
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

			/*
				Todo: when considering traffic lights it could be necessary not only checking current TL phase
				but also consider whether agent has already passed "its" stop line, that is even if TL is read but agent has already
				slipped over the stop line but will keep going
			*/

			if	(	crossTraffic.Paths.Num() > 0
				&&	(agentDirectionMask & maneuverMask) != 0
				&&	(	ignoreTrafficLights
					||	crossTraffic.TrafficLight == 0
					||	crossTraffic.TrafficLight->CurrentPhase != EDeepDriveTrafficLightPhase::RED
					)
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
