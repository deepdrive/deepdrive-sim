
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

uint32 SDeepDriveRoadNetwork::findClosestLink(const FVector &pos) const
{
	uint32 linkInd = 0;

	float bestDist = TNumericLimits<float>::Max();
	for(auto &link : Links)
	{
		for(auto &lane : link.Value.Lanes)
		{
			for(auto &segmentId : lane.Segments)
			{
				const SDeepDriveRoadSegment &curSegment = Segments[segmentId];
				float dist = (curSegment.StartPoint - pos).Size();
				if(dist < bestDist)
				{
					linkInd = curSegment.LinkId;
					bestDist = dist;
				}
				dist = (curSegment.EndPoint - pos).Size();
				if(dist < bestDist)
				{
					linkInd = curSegment.LinkId;
					bestDist = dist;
				}
			}
		}
	}

	return linkInd;
}

uint32 SDeepDriveRoadNetwork::findClosestSegment(const FVector &pos, EDeepDriveLaneType laneType) const
{
	uint32 key = 0;
	float dist = TNumericLimits<float>::Max();

	for (auto curIt = Segments.CreateConstIterator(); curIt; ++curIt)
	{
		auto segment = curIt.Value();
		if(segment.LaneType == laneType)
		{
			if(segment.SplineCurves.Position.Points.Num() > 0)
			{
				float curDist = (segment.StartPoint - pos).Size();
				if(curDist < dist)
				{
					dist = curDist;
					key = curIt.Key();
				}
				curDist = (segment.EndPoint - pos).Size();
				if(curDist < dist)
				{
					dist = curDist;
					key = curIt.Key();
				}
			}
			else
			{
				FVector cp = FMath::ClosestPointOnLine(segment.StartPoint, segment.EndPoint, pos);
				const float curDist = (cp - pos).Size();
				if(curDist < dist)
				{
					dist = curDist;
					key = curIt.Key();
				}
			}

		}
	}

	return key;
}
