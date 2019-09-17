
#include "Simulation/Traffic/Path/DeepDriveManeuverCrossTrafficExtractor.h"
#include "Simulation/Traffic/Path/DeepDrivePathDefines.h"

DEFINE_LOG_CATEGORY(LogDeepDriveManeuverCrossTrafficExtractor);

DeepDriveManeuverCrossTrafficExtractor::DeepDriveManeuverCrossTrafficExtractor(const SDeepDriveRoadNetwork &roadNetwork)
	:	m_RoadNetwork(roadNetwork)
{

}

void DeepDriveManeuverCrossTrafficExtractor::extract(SDeepDriveManeuver &maneuver)
{
	switch(maneuver.JunctionType)
	{
		case EDeepDriveJunctionType::FOUR_WAY_JUNCTION:
			extractFromCrossJunction(maneuver);
			break;
	}
}


void DeepDriveManeuverCrossTrafficExtractor::extractFromCrossJunction(SDeepDriveManeuver &maneuver)
{
	const uint32 fromLinkId = maneuver.FromLinkId;
	const uint32 toLinkId = maneuver.ToLinkId;
	const SDeepDriveRoadLink &fromLink = m_RoadNetwork.Links[fromLinkId];
	const SDeepDriveRoadLink &toLink = m_RoadNetwork.Links[maneuver.ToLinkId];
	const SDeepDriveJunction &junction = m_RoadNetwork.Junctions[fromLink.ToJunctionId];

	UE_LOG(LogDeepDriveManeuverCrossTrafficExtractor, Log, TEXT("Extracting cross traffic for Cross Junction"));
#if 0
	TSortedEntries sortedEntries;

	for(const SDeepDriveJunctionEntry &entry : junction.Entries)
	{
		if(fromLinkId != entry.LinkId)
		{
			const SDeepDriveRoadLink &entryLink = m_RoadNetwork.Links[entry.LinkId];

			float curRelDir = getRelativeDirection(fromLink, entryLink);
			sortedEntries.Add( TPair<float, const SDeepDriveJunctionEntry*>(curRelDir, &entry));
		}
	}

	// sort entries counter clockwise (from right to left)
	if(sortedEntries[1].Key < sortedEntries[0].Key)
		sortedEntries.Swap(0, 1);
	if(sortedEntries[2].Key < sortedEntries[1].Key)
	{
		sortedEntries.Swap(1, 2);
		if(sortedEntries[1].Key < sortedEntries[0].Key)
			sortedEntries.Swap(0, 1);
	}

	for(auto elem : sortedEntries)
		UE_LOG(LogDeepDriveManeuverCrossTrafficExtractor, Log, TEXT("link %d relDir %f"), elem.Value->LinkId, elem.Key);

	// first check road from right
	switch(maneuver.ManeuverType)
	{
		case EDeepDriveManeuverType::TURN_RIGHT:
			break;

		case EDeepDriveManeuverType::GO_ON_STRAIGHT:
			if ((fromLink.RoadPriority == EDeepDriveRoadPriority::MAIN_ROAD && m_RoadNetwork.Links[sortedEntries[0].Value->LinkId].RoadPriority == EDeepDriveRoadPriority::MAIN_ROAD)

			)
			{
				maneuver.CrossTrafficRoads.Add(SDeepDriveCrossTrafficRoad(fromLinkId, sortedEntries[0].Value->LinkId, 1500.0f, 1000.0f));
			}
			break;

		case EDeepDriveManeuverType::TURN_LEFT:
			break;
	}

	// now a bunch of hand picked checks to identify cross traffic relevant roads

	if(fromLink.RoadPriority == EDeepDriveRoadPriority::MAIN_ROAD)
	{
		if(toLink.RoadPriority == EDeepDriveRoadPriority::MINOR_ROAD)
		{
			if(maneuver.ManeuverType == EDeepDriveManeuverType::GO_ON_STRAIGHT)
			{
				// if road to right is main road, add as cross traffic road
				if(m_RoadNetwork.Links[sortedEntries[0].Value->LinkId].RoadPriority == EDeepDriveRoadPriority::MAIN_ROAD)
					maneuver.CrossTrafficRoads.Add(SDeepDriveCrossTrafficRoad(fromLinkId, sortedEntries[0].Value->LinkId, 1500.0f, 1000.0f));
			}
		}
	}
	else
	{
		switch(maneuver.ManeuverType)
		{
			case EDeepDriveManeuverType::TURN_RIGHT:
				if(m_RoadNetwork.Links[sortedEntries[1].Value->LinkId].RoadPriority == EDeepDriveRoadPriority::MAIN_ROAD)
					maneuver.CrossTrafficRoads.Add(SDeepDriveCrossTrafficRoad(fromLinkId, sortedEntries[1].Value->LinkId, 1500.0f, 1000.0f));
				if(m_RoadNetwork.Links[sortedEntries[2].Value->LinkId].RoadPriority == EDeepDriveRoadPriority::MAIN_ROAD)
					maneuver.CrossTrafficRoads.Add(SDeepDriveCrossTrafficRoad(sortedEntries[2].Value->LinkId, toLinkId, 1500.0f, 1000.0f));
				break;

			case EDeepDriveManeuverType::GO_ON_STRAIGHT:
				break;

			case EDeepDriveManeuverType::TURN_LEFT:
				break;
		}
	}
	
		for(auto crt : maneuver.CrossTrafficRoads)
			UE_LOG(LogDeepDriveManeuverCrossTrafficExtractor, Log, TEXT("CrossTraffic from %d to %d"), crt.FromLinkId, crt.ToLinkId);

#endif
}
#if 0
void DeepDriveManeuverCrossTrafficExtractor::addCrossRoadsForLink(const SDeepDriveJunction &junction, uint32 fromLinkId)
{
	const SDeepDriveJunctionEntry *junctionEntry = 0;
	if (findJunctionEntry(fromLinkId, junctionEntry))
	{

	}
}
#endif

float DeepDriveManeuverCrossTrafficExtractor::getRelativeDirection(const SDeepDriveRoadLink &fromLink, const SDeepDriveRoadLink &outLink)
{
	FVector2D fromDir(fromLink.EndDirection);
	FVector2D outDir(-outLink.StartDirection);

	fromDir.Normalize();
	outDir.Normalize();

	FVector2D fromNrm(-fromDir.Y, fromDir.X);

	float relDir = FVector2D::DotProduct(fromDir, outDir);
	if(FVector2D::DotProduct(fromNrm, outDir) >= 0.0f)
	{
		relDir += 1.0f;
	}
	else
	{
		relDir = 3.0f - relDir;
	}
	
	return relDir;
}
