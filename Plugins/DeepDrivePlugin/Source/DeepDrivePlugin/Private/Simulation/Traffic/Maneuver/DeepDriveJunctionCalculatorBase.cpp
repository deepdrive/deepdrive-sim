
#include "Simulation/Traffic/Maneuver/DeepDriveJunctionCalculatorBase.h"
#include "Simulation/RoadNetwork/DeepDriveRoadNetwork.h"

DeepDriveJunctionCalculatorBase::DeepDriveJunctionCalculatorBase(const SDeepDriveRoadNetwork &roadNetwork)
	:	m_RoadNetwork(roadNetwork)
{
}

float DeepDriveJunctionCalculatorBase::getRelativeDirection(const SDeepDriveRoadLink &fromLink, const SDeepDriveRoadLink &outLink)
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

DeepDriveJunctionCalculatorBase::TSortedJunctionEntries DeepDriveJunctionCalculatorBase::collectJunctionEntries(const SDeepDriveJunction &junction, const SDeepDriveRoadLink &fromLink)
{
	uint32 fromLinkId = fromLink.LinkId;
	TSortedJunctionEntries sortedEntries;
	for(const SDeepDriveJunctionEntry &entry : junction.Entries)
	{
		if(fromLinkId != entry.LinkId)
		{
			const SDeepDriveRoadLink &entryLink = m_RoadNetwork.Links[entry.LinkId];

			float curRelDir = getRelativeDirection(fromLink, entryLink);
			sortedEntries.Add( TJunctionEntryItem(curRelDir, &entry));
		}
	}
	return sortedEntries;
}
