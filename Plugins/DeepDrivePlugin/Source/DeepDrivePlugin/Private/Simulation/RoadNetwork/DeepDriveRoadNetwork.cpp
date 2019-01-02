
#include "Public/Simulation/RoadNetwork/DeepDriveRoadNetwork.h"


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

float SDeepDriveRoadNetwork::getSpeedLimit(uint32 segmentId, float relativePos) const
{
	const SDeepDriveRoadSegment &segment = Segments[segmentId];

	float speedLimit = -1.0f;

	if (segment.SpeedLimits.Num() > 0)
	{
		for (signed i = 0; i < segment.SpeedLimits.Num(); ++i)
		{
			if (relativePos >= segment.SpeedLimits[i].X)
			{
				speedLimit = segment.SpeedLimits[i].Y;
			}
			else
				break;
		}
	}

	return speedLimit;
}
