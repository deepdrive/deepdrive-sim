
#include "Public/Simulation/RoadNetwork/DeepDriveRoadNetwork.h"

int32 SDeepDriveRoadLink::getRightMostLane(EDeepDriveLaneType type) const
{
	int32 curLaneInd = Lanes.Num() - 1;
	while(curLaneInd >= 0)
	{
		if(Lanes[curLaneInd].LaneType == type)
			break;
		
		--curLaneInd;
	}

	return curLaneInd;
}

uint32 SDeepDriveJunction::findConnectionSegment(uint32 fromSegment, uint32 toSegment) const
{
	uint32 connectionSegment = 0;

	for (auto &connection : Connections)
	{
		if(connection.FromSegment == fromSegment && connection.ToSegment == toSegment)
		{
			connectionSegment = connection.ConnectionSegment;
			break;
		}
	}

	return connectionSegment;
}
