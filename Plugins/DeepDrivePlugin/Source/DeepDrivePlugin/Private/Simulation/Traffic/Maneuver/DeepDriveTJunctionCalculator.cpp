#include "Simulation/Traffic/Maneuver/DeepDriveTJunctionCalculator.h"
#include "Simulation/Traffic/Path/DeepDrivePathDefines.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoadNetwork.h"

DEFINE_LOG_CATEGORY(LogDeepDriveTJunctionCalculator);

DeepDriveTJunctionCalculator::DeepDriveTJunctionCalculator(const SDeepDriveRoadNetwork &roadNetwork)
	:	DeepDriveFourWayJunctionCalculator(roadNetwork)
{

}

void DeepDriveTJunctionCalculator::calculate(SDeepDriveManeuver &maneuver)
{
	const uint32 junctionSubType = static_cast<uint32_t> (maneuver.JunctionSubType);

	if(createBehaviorTree(maneuver))
	{
		UE_LOG(LogDeepDriveTJunctionCalculator, Log, TEXT("Behavior tree created"));

		const SDeepDriveRoadLink &fromLink = m_RoadNetwork.Links[maneuver.FromLinkId];
		const SDeepDriveJunction &junction = m_RoadNetwork.Junctions[fromLink.ToJunctionId];

		// sort entries counter clockwise (from right to left)
		TSortedJunctionEntries sortedEntries = collectJunctionEntries(junction, fromLink);
		if(sortedEntries[1].Key < sortedEntries[0].Key)
			sortedEntries.Swap(0, 1);

		Type type = deduceJunctionType(sortedEntries);
		UE_LOG(LogDeepDriveTJunctionCalculator, Log, TEXT("Junction Type %d"), static_cast<int32> (type) );

		switch(type)
		{
			case T_Junction:
				extractCrossRoadTraffic(sortedEntries[0].Value->LinkId, maneuver, junction, m_CrossRoadRightMasks);
				extractCrossRoadTraffic(sortedEntries[1].Value->LinkId, maneuver, junction, m_CrossRoadLeftMasks);
				break;

			case Right:
				extractCrossRoadTraffic(sortedEntries[0].Value->LinkId, maneuver, junction, m_CrossRoadRightMasks);
				extractCrossRoadTraffic(sortedEntries[1].Value->LinkId, maneuver, junction, m_CrossRoadStraightMasks);
				break;
			case Left:
				extractCrossRoadTraffic(sortedEntries[0].Value->LinkId, maneuver, junction, m_CrossRoadStraightMasks);
				extractCrossRoadTraffic(sortedEntries[1].Value->LinkId, maneuver, junction, m_CrossRoadLeftMasks);
				break;
		}
	}
}

DeepDriveTJunctionCalculator::Type DeepDriveTJunctionCalculator::deduceJunctionType(TSortedJunctionEntries &sortedEntries)
{
	if(sortedEntries[0].Key > 0.5f && sortedEntries[0].Key < 1.5f)
	{
		return sortedEntries[1].Key > 2.5f && sortedEntries[1].Key < 3.5f ? T_Junction : Right;
	}
	else
		return Left;
	
}
