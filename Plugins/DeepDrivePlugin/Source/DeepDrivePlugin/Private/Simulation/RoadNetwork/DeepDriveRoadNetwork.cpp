
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
				FVector closestPos = curSegment.findClosestPoint(pos);
				const float curDist = (closestPos - pos).Size();
				if (curDist < bestDist)
				{
					bestDist = curDist;
					linkInd = curSegment.LinkId;
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
			const SDeepDriveRoadSegment &curSegment = curIt.Value();
			FVector closestPos = curSegment.findClosestPoint(pos);
			const float curDist = (closestPos - pos).Size();
			if(curDist < dist)
			{
				dist = curDist;
				key = curIt.Key();
			}
		}
	}

	return key;
}

FVector SDeepDriveRoadSegment::findClosestPoint(const FVector &pos) const
{
	FVector closestPos;
	if (SplineCurves.Position.Points.Num() > 0)
	{
		const FVector localPos = Transform.InverseTransformPosition(pos);
		float dummy;
		const float key = SplineCurves.Position.InaccurateFindNearest(localPos, dummy);
		closestPos = SplineCurves.Position.Eval(key, FVector::ZeroVector);
		closestPos = Transform.TransformPosition(closestPos);
	}
	else
	{
		closestPos = FMath::ClosestPointOnLine(StartPoint, EndPoint, pos);
	}
	return closestPos;
}