
#include "Public/Simulation/RoadNetwork/DeepDriveRoadNetwork.h"

DEFINE_LOG_CATEGORY(LogDeepDriveRoadNetwork);

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

ADeepDriveTrafficLight* SDeepDriveJunctionEntry::getTrafficLight(uint32 toLinkId) const
{
	for(auto &turnDefinition : TurnDefinitions)
	{
		if(turnDefinition.ToLinkId == toLinkId)
			return turnDefinition.TrafficLight;
	}
	return 0;
}

bool SDeepDriveJunction::findJunctionConnection(uint32 fromLinkId, uint32 fromSegment, uint32 toSegment, SDeepDriveJunctionConnection &junctionConnection) const
{
	for (auto &entry : Entries)
	{
		if	(	fromLinkId == 0
			||	entry.LinkId == fromLinkId
			)
		{
			for(auto &connection : entry.Connections)
			{
				if(connection.SegmentId == fromSegment && connection.ToSegmentId == toSegment)
				{
					junctionConnection = connection;
					return true;
				}
			}
		}
	}

	return false;
}

int32 SDeepDriveJunction::findJunctionEntry(uint32 fromLinkId, const SDeepDriveJunctionEntry* &junctionEntry) const
{
	int32 index = 0;
	for (auto &entry : Entries)
	{
		if(entry.LinkId == fromLinkId)
		{
			junctionEntry = &entry;
			return index;
		}
		index++;
	}

	return -1;
}

bool SDeepDriveJunction::isTurningAllowed(uint32 fromLinkId, uint32 toLinkId) const
{
	for (auto &entry : Entries)
	{
		if(entry.LinkId == fromLinkId)
		{
			for(auto &turnDefinition : entry.TurnDefinitions)
			{
				if (turnDefinition.ToLinkId == toLinkId)
					return turnDefinition.ManeuverType != EDeepDriveManeuverType::UNDEFINED;
			}
		}
	}

	return false;
}

EDeepDriveManeuverType SDeepDriveJunction::getManeuverType(uint32 fromLinkId, uint32 toLinkId) const
{
	for (auto &entry : Entries)
	{
		if(entry.LinkId == fromLinkId)
		{
			for(auto &turnDefinition : entry.TurnDefinitions)
			{
				if (turnDefinition.ToLinkId == toLinkId)
					return turnDefinition.ManeuverType;
			}
		}
	}

	return EDeepDriveManeuverType::UNDEFINED;
}

void SDeepDriveJunction::getRelevantAgents(uint32 fromLinkId, uint32 toLinkId, ADeepDriveAgent *egoAgent, TArray<ADeepDriveAgent *> &agents) const
{
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
	FVector closestPos;

	for (auto curIt = Segments.CreateConstIterator(); curIt; ++curIt)
	{
		auto segment = curIt.Value();
		if(segment.LaneType == laneType)
		{
			const SDeepDriveRoadSegment &curSegment = curIt.Value();
			closestPos = curSegment.findClosestPoint(pos);
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

FVector SDeepDriveRoadNetwork::getLocationOnLink(uint32 linkId, EDeepDriveLaneType laneType, float t) const
{
	FVector location = FVector::ZeroVector;
	if(linkId)
	{
		const SDeepDriveRoadLink &link = Links[linkId];
		int32 laneInd = link.getRightMostLane(laneType);
		if(laneInd >= 0 && link.Lanes[laneInd].Segments.Num() > 0)
		{
			location = Segments[link.Lanes[laneInd].Segments[0]].getLocationOnSegment(t);
		}
	}

	return location;
}

FString SDeepDriveRoadNetwork::getDebugLinkName(uint32 linkId) const
{
	FString name = LinkNameMap.Contains(linkId) ? LinkNameMap[linkId] : "Unknown Link";

	return "(" + FString::FromInt(linkId) + ") " + name;
}

FVector SDeepDriveRoadSegment::findClosestPoint(const FVector &pos) const
{
	FVector closestPos;
	if (hasSpline())
	{
		const FVector localPos = SplineTransform.InverseTransformPosition(pos);
		float dummy;
		const float key = SplineCurves.Position.InaccurateFindNearest(localPos, dummy);
		closestPos = SplineCurves.Position.Eval(key, FVector::ZeroVector);
		closestPos = SplineTransform.TransformPosition(closestPos);
	}
	else
	{
		closestPos = FMath::ClosestPointOnLine(StartPoint, EndPoint, pos);
	}
	return closestPos;
}

float SDeepDriveRoadSegment::getHeading(const FVector &pos) const
{
	float heading = Heading;
	if (hasSpline())
	{
		const FVector localPos = SplineTransform.InverseTransformPosition(pos);
		float dummy;
		const float key = SplineCurves.Position.InaccurateFindNearest(localPos, dummy);
		FVector p0 = SplineCurves.Position.Eval(key, FVector::ZeroVector);
		p0 = SplineTransform.TransformPosition(p0);
		FVector p1 = SplineCurves.Position.Eval(key * 1.01f, FVector::ZeroVector);
		p1 = SplineTransform.TransformPosition(p1);

		FVector2D dir = FVector2D(p1 - p0);
		dir.Normalize();
		heading = FMath::RadiansToDegrees(FMath::Atan2(dir.Y, dir.X));
	}
	return heading;
}

FVector SDeepDriveRoadSegment::getLocationOnSegment(float relativePos) const
{
	relativePos = FMath::Clamp(relativePos, 0.0f, 1.0f);
	FVector location;
	if(hasSpline())
	{
		const float splineLength = SplineCurves.GetSplineLength();
		const float key = SplineCurves.ReparamTable.Eval(relativePos * splineLength, 0.0f);

		location = SplineCurves.Position.Eval(key, FVector::ZeroVector);
		location = SplineTransform.TransformPosition(location);
	}
	else
	{
		location = FMath::Lerp(StartPoint, EndPoint, relativePos);
	}

	return location;
}

