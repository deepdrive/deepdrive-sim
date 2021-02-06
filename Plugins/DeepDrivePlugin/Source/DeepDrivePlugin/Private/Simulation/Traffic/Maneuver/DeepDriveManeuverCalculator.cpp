
#include "Simulation/Traffic/Maneuver/DeepDriveManeuverCalculator.h"
#include "Simulation/Traffic/Path/DeepDrivePathDefines.h"

#include "Simulation/Traffic/Maneuver/DeepDriveFourWayJunctionCalculator.h"
#include "Simulation/Traffic/Maneuver/DeepDriveTJunctionCalculator.h"

#include "Simulation/RoadNetwork/DeepDriveRoadNetwork.h"

#include "Simulation/Traffic/BehaviorTree/DeepDriveBehaviorTreeFactory.h"

#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBehaviorTree.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBehaviorTreeNode.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBlackboard.h"

#include "Simulation/Traffic/BehaviorTree/Tasks/DeepDriveTBTStopAtLocationTask.h"
#include "Simulation/Traffic/BehaviorTree/Tasks/DeepDriveTBTHasPassedLocationTask.h"

#include "Simulation/Agent/DeepDriveAgent.h"

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
			UE_LOG(LogDeepDriveManeuverCalculator, Log, TEXT("Calc maneuver from %s to %s"), *(m_RoadNetwork.getDebugLinkName(fromLinkId)), *(m_RoadNetwork.getDebugLinkName(toLinkId)) );

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
					blackboard.setIntegerValue("AgentId", agent.GetAgentId());

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
						UE_LOG(LogDeepDriveManeuverCalculator, Log, TEXT("CrossTraffic from %s to %s as %d"), *(m_RoadNetwork.getDebugLinkName(crt.FromLinkId)), *(m_RoadNetwork.getDebugLinkName(crt.ToLinkId)), static_cast<int32> (crt.ManeuverType));
				}
			}

			route.Maneuvers.Add(maneuver);
		}
	}
	addFinalManeuver(route, agent);
}

void DeepDriveManeuverCalculator::addFinalManeuver(SDeepDriveRoute &route, ADeepDriveAgent &agent)
{
	TArray<uint32> &routeLinks = route.Links;
	const SDeepDriveRoadLink &finalLink = m_RoadNetwork.Links[routeLinks.Last()];

	SDeepDriveManeuver maneuver;
	maneuver.FromLinkId = finalLink.LinkId;
	maneuver.ToLinkId = finalLink.LinkId;

	maneuver.JunctionType = EDeepDriveJunctionType::DESTINATION_REACHED;

	maneuver.EntryPoint = finalLink.StartPoint;
	maneuver.ExitPoint = route.Destination;

	DeepDriveTrafficBehaviorTree *behaviorTree = new DeepDriveTrafficBehaviorTree("StopAtEnd");
	if (behaviorTree)
	{
		maneuver.BehaviorTree = behaviorTree;

		DeepDriveTrafficBehaviorTreeNode *stopAtNode = behaviorTree->createNode(0);

		stopAtNode->addTask(new DeepDriveTBTStopAtLocationTask("DestinationLocation", 0.25f, 0.0f));
		stopAtNode->addTask(new DeepDriveTBTHasPassedLocationTask("DestinationLocation", 0.0f, "DestinationReached"));

		DeepDriveTrafficBlackboard &blackboard = behaviorTree->getBlackboard();
		blackboard.setSimulation(m_Simulation);
		blackboard.setAgent(agent);
		blackboard.setIntegerValue("AgentId", agent.GetAgentId());
		blackboard.setVectorValue("DestinationLocation", route.Destination);

		route.Maneuvers.Add(maneuver);
	}
}
