
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
					//	check if agent close to this path
					for (auto &curPathPoint : curPath)
					{
						if (FVector::DistSquared(curAgentLoc, curPathPoint.Location) < (10000.0f))
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
