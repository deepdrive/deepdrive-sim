
#include "Simulation/Traffic/Maneuver/DeepDriveManeuverCalculator.h"
#include "Simulation/Traffic/Path/DeepDrivePathDefines.h"

#include "Simulation/Traffic/Maneuver/DeepDriveFourWayJunctionCalculator.h"
#include "Simulation/Traffic/Maneuver/DeepDriveTJunctionCalculator.h"

#include "Public/Simulation/RoadNetwork/DeepDriveRoadNetwork.h"

#include "Simulation/Traffic/BehaviorTree/DeepDriveBehaviorTreeFactory.h"

#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBehaviorTree.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBehaviorTreeNode.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBlackboard.h"

DEFINE_LOG_CATEGORY(LogDeepDriveManeuverCalculator);

DeepDriveManeuverCalculator::DeepDriveManeuverCalculator(const SDeepDriveRoadNetwork &roadNetwork, ADeepDriveSimulation &simulation)
	:	m_RoadNetwork(roadNetwork)
	,	m_Simulation(simulation)
{
	m_JunctionCalculators.Add(static_cast<uint32>(EDeepDriveJunctionType::FOUR_WAY_JUNCTION), new DeepDriveFourWayJunctionCalculator(roadNetwork));
	m_JunctionCalculators.Add(static_cast<uint32>(EDeepDriveJunctionType::T_JUNCTION), new DeepDriveTJunctionCalculator(roadNetwork));
}

void DeepDriveManeuverCalculator::calculate(SDeepDriveRoute &route, ADeepDriveAgent &agent)
{
	TArray<uint32> &routeLinks = route.Links;

	for (int32 i = 1; i < routeLinks.Num(); ++i)
	{
		const SDeepDriveRoadLink &fromLink = m_RoadNetwork.Links[routeLinks[i - 1]];
		const SDeepDriveRoadLink &toLink = m_RoadNetwork.Links[routeLinks[i]];
		const SDeepDriveJunction &junction = m_RoadNetwork.Junctions[fromLink.ToJunctionId];
		const uint32 fromLinkId = fromLink.LinkId;
		const uint32 toLinkId = toLink.LinkId;

		const SDeepDriveJunctionEntry *junctionEntry = 0;
		if(junction.findJunctionEntry(fromLinkId, junctionEntry))
		{
			UE_LOG(LogDeepDriveManeuverCalculator, Log, TEXT("Calc maneuver from %d to %d"), fromLinkId, toLinkId );

			SDeepDriveManeuver maneuver;
			maneuver.JunctionType = junction.JunctionType;
			maneuver.JunctionSubType = junctionEntry->JunctionSubType;
			maneuver.ManeuverType = junction.getManeuverType(fromLinkId, toLinkId);
			maneuver.FromLinkId = fromLinkId;
			maneuver.ToLinkId = toLinkId;
			maneuver.RightOfWay = junctionEntry->RightOfWay;
			maneuver.FromRoadPriority = fromLink.RoadPriority;
			maneuver.ToRoadPriority = toLink.RoadPriority;

			maneuver.EntryPoint = junctionEntry->ManeuverEntryPoint;
			maneuver.ExitPoint = m_RoadNetwork.Links[maneuver.ToLinkId].StartPoint;

			maneuver.TrafficLight = junctionEntry->getTrafficLight(toLink.LinkId);

			const uint32 junctionType = static_cast<uint32>(maneuver.JunctionType);
			if(m_JunctionCalculators.Contains(junctionType))
			{
				m_JunctionCalculators[junctionType]->calculate(maneuver);

				if (maneuver.BehaviorTree)
				{
					DeepDriveTrafficBlackboard &blackboard = maneuver.BehaviorTree->getBlackboard();
					blackboard.setSimulation(m_Simulation);
					blackboard.setAgent(agent);

					blackboard.setVectorValue("StopLineLocation", fromLink.StopLineLocation);
					blackboard.setVectorValue("LineOfSightLocation", junctionEntry->LineOfSight);

					for(auto &turnDef : junctionEntry->TurnDefinitions)
					{
						if (turnDef.ToLinkId == toLinkId)
						{
							blackboard.setVectorValue("WaitingLocation", turnDef.WaitingLocation);
							break;
						}
					}

					for(auto crt : maneuver.CrossTrafficRoads)
						UE_LOG(LogDeepDriveManeuverCalculator, Log, TEXT("CrossTraffic from %d to %d"), crt.FromLinkId, crt.ToLinkId);

				}
			}

			route.Maneuvers.Add(maneuver);
		}
	}
}
