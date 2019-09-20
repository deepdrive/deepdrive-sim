
#include "Simulation/Traffic/Maneuver/DeepDriveFourWayJunctionCalculator.h"

#include "Simulation/Traffic/Path/DeepDrivePathDefines.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoadNetwork.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveBehaviorTreeFactory.h"

#include "Simulation/TrafficLight/DeepDriveTrafficLight.h"

DEFINE_LOG_CATEGORY(LogDeepDriveFourWayJunctionCalculator);

DeepDriveFourWayJunctionCalculator::DeepDriveFourWayJunctionCalculator(const SDeepDriveRoadNetwork &roadNetwork)
	:	DeepDriveJunctionCalculatorBase(roadNetwork)
{
	loadConfiguration("");
}

void DeepDriveFourWayJunctionCalculator::calculate(SDeepDriveManeuver &maneuver)
{
	const uint32 junctionSubType = static_cast<uint32_t> (maneuver.JunctionSubType);

	if(createBehaviorTree(maneuver))
	{
		UE_LOG(LogDeepDriveFourWayJunctionCalculator, Log, TEXT("Behavior tree created"));

		const SDeepDriveRoadLink &fromLink = m_RoadNetwork.Links[maneuver.FromLinkId];
		const SDeepDriveJunction &junction = m_RoadNetwork.Junctions[fromLink.ToJunctionId];

		TSortedJunctionEntries sortedEntries = collectJunctionEntries(junction, fromLink);

		// sort entries counter clockwise (from right to left)
		if(sortedEntries[1].Key < sortedEntries[0].Key)
			sortedEntries.Swap(0, 1);
		if(sortedEntries[2].Key < sortedEntries[1].Key)
		{
			sortedEntries.Swap(1, 2);
			if(sortedEntries[1].Key < sortedEntries[0].Key)
				sortedEntries.Swap(0, 1);
		}

		extractCrossRoadTraffic(sortedEntries[0].Value->LinkId, maneuver, junction, m_CrossRoadRightMasks);
		extractCrossRoadTraffic(sortedEntries[1].Value->LinkId, maneuver, junction, m_CrossRoadStraightMasks);
		extractCrossRoadTraffic(sortedEntries[2].Value->LinkId, maneuver, junction, m_CrossRoadLeftMasks);
	}
}

bool DeepDriveFourWayJunctionCalculator::createBehaviorTree(SDeepDriveManeuver &maneuver)
{
	if(maneuver.TrafficLight && maneuver.TrafficLight->IsActive())
	{
		UE_LOG(LogDeepDriveFourWayJunctionCalculator, Log, TEXT("Caluclate 4-way junction with active traffic light ") );

		DeepDriveBehaviorTreeFactory &factory = DeepDriveBehaviorTreeFactory::GetInstance();

		maneuver.BehaviorTree = 	maneuver.ManeuverType == EDeepDriveManeuverType::TURN_LEFT
								?	factory.createBehaviorTree("four_way_tl_upl")
								:	factory.createBehaviorTree("four_way_tl_rs");
	}
	else
	{
		const uint32 junctionSubType = static_cast<uint32_t> (maneuver.JunctionSubType);
		const uint32 key = calcBehaviorTreeKey(maneuver);
		const FString treeId = m_BehaviorTreeIds.Contains(key) ? m_BehaviorTreeIds[key] : "";

		UE_LOG(LogDeepDriveFourWayJunctionCalculator, Log, TEXT("Calculate 4-way junction with sub type %d and maneuver type %d key 0x%x treeId %s"), junctionSubType, static_cast<uint32> (maneuver.ManeuverType), key, *(treeId) );

		maneuver.BehaviorTree =		treeId.IsEmpty() == false
								?	DeepDriveBehaviorTreeFactory::GetInstance().createBehaviorTree(m_BehaviorTreeIds[key])
								:	0
								;
	}

	return maneuver.BehaviorTree != 0;
}

void DeepDriveFourWayJunctionCalculator::extractCrossRoadTraffic(uint32 curEntryLinkId, SDeepDriveManeuver &maneuver, const SDeepDriveJunction &junction, const TCrossTrafficRoadMasks &masks)
{
	const uint32 junctionSubType = static_cast<uint32_t> (maneuver.JunctionSubType);
	if (masks.Contains(junctionSubType))
	{
		uint32 mask = 0;
		switch (maneuver.ManeuverType)
		{
		case EDeepDriveManeuverType::TURN_RIGHT:
			mask = masks[junctionSubType].turn_right;
			break;

		case EDeepDriveManeuverType::GO_ON_STRAIGHT:
			mask = masks[junctionSubType].go_on_straight;
			break;

		case EDeepDriveManeuverType::TURN_LEFT:
			mask = masks[junctionSubType].turn_left;
			break;
		}

		UE_LOG(LogDeepDriveFourWayJunctionCalculator, Log, TEXT("Checking cross road with id %d and mask %d"), curEntryLinkId, mask);
		for (auto &entry : junction.Entries)
		{
			if (entry.LinkId == curEntryLinkId)
			{
				for (auto &turnDef : entry.TurnDefinitions)
				{
					if ((mask & (1 << static_cast<uint32>(turnDef.ManeuverType))) != 0)
					{
						maneuver.CrossTrafficRoads.Add(SDeepDriveCrossTrafficRoad(curEntryLinkId, turnDef.ToLinkId, 2500.0f, 1000.0f));
					}
				}
			}
		}
	}
}

void DeepDriveFourWayJunctionCalculator::loadConfiguration(const FString &configFile)
{
	m_BehaviorTreeIds.Add(calcBehaviorTreeKey(1, 3), "four_way_aws");		// all-way stop
	m_BehaviorTreeIds.Add(calcBehaviorTreeKey(1, 4), "four_way_rbl");		// right before left
	m_BehaviorTreeIds.Add(calcBehaviorTreeKey(2, 1), "four_way_yield");		// yield
	m_BehaviorTreeIds.Add(calcBehaviorTreeKey(2, 2), "four_way_stop");		// stop
	m_BehaviorTreeIds.Add(calcBehaviorTreeKey(3, 0), "four_way_row");		// right of way
	m_BehaviorTreeIds.Add(calcBehaviorTreeKey(4, 0), "four_way_row");		// right of way
	m_BehaviorTreeIds.Add(calcBehaviorTreeKey(5, 0), "four_way_row");		// right of way
	m_BehaviorTreeIds.Add(calcBehaviorTreeKey(6, 1), "four_way_yield");		// yield
	m_BehaviorTreeIds.Add(calcBehaviorTreeKey(6, 2), "four_way_stop");		// stop
	m_BehaviorTreeIds.Add(calcBehaviorTreeKey(7, 1), "four_way_yield");		// yield
	m_BehaviorTreeIds.Add(calcBehaviorTreeKey(7, 2), "four_way_stop");		// stop

	m_CrossRoadRightMasks.Add(1, SCrossTrafficRoadMask(0, 14, 12));
	m_CrossRoadRightMasks.Add(2, SCrossTrafficRoadMask(0, 14, 12));
	m_CrossRoadRightMasks.Add(3, SCrossTrafficRoadMask(0, 0, 0));
	m_CrossRoadRightMasks.Add(4, SCrossTrafficRoadMask(0, 14, 12));
	m_CrossRoadRightMasks.Add(5, SCrossTrafficRoadMask(0, 0, 0));
	m_CrossRoadRightMasks.Add(6, SCrossTrafficRoadMask(0, 14, 12));
	m_CrossRoadRightMasks.Add(7, SCrossTrafficRoadMask(0, 14, 12));

	m_CrossRoadStraightMasks.Add(1, SCrossTrafficRoadMask(0, 0, 0));
	m_CrossRoadStraightMasks.Add(2, SCrossTrafficRoadMask(0, 0, 6));
	m_CrossRoadStraightMasks.Add(3, SCrossTrafficRoadMask(0, 0, 6));
	m_CrossRoadStraightMasks.Add(4, SCrossTrafficRoadMask(0, 0, 0));
	m_CrossRoadStraightMasks.Add(5, SCrossTrafficRoadMask(0, 0, 0));
	m_CrossRoadStraightMasks.Add(6, SCrossTrafficRoadMask(0, 8, 6));
	m_CrossRoadStraightMasks.Add(7, SCrossTrafficRoadMask(8, 8, 6));

	m_CrossRoadLeftMasks.Add(1, SCrossTrafficRoadMask(0, 0, 0));
	m_CrossRoadLeftMasks.Add(2, SCrossTrafficRoadMask(4, 12, 12));
	m_CrossRoadLeftMasks.Add(3, SCrossTrafficRoadMask(0, 0, 0));
	m_CrossRoadLeftMasks.Add(4, SCrossTrafficRoadMask(0, 0, 0));
	m_CrossRoadLeftMasks.Add(5, SCrossTrafficRoadMask(0, 0, 0));
	m_CrossRoadLeftMasks.Add(6, SCrossTrafficRoadMask(4, 12, 12));
	m_CrossRoadLeftMasks.Add(7, SCrossTrafficRoadMask(0, 0, 0));
}

uint32 DeepDriveFourWayJunctionCalculator::calcBehaviorTreeKey(const SDeepDriveManeuver &maneuver) const
{
	return calcBehaviorTreeKey(static_cast<uint32> (maneuver.JunctionSubType), static_cast<uint32> (maneuver.RightOfWay));
}

uint32 DeepDriveFourWayJunctionCalculator::calcBehaviorTreeKey(uint32 junctionSubType, uint32 rightOfWay) const
{
	uint32 key = 0;

	key = (junctionSubType & 0xFF) | ((rightOfWay & 0x7) << 8);

	// UE_LOG(LogDeepDriveFourWayJunctionCalculator, Log, TEXT("Key for %d %d -> %d"), junctionSubType, rightOfWay, key );
	return key;
}
