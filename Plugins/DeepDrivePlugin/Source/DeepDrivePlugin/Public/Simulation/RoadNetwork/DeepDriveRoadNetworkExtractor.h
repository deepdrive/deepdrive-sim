
#pragma once

#include "CoreMinimal.h"

#include "Public/Simulation/RoadNetwork/DeepDriveRoadNetwork.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveRoadNetworkExtractor, Log, All);

class ADeepDriveRoadLinkProxy;
class ADeepDriveRoadSegmentProxy;

class DeepDriveRoadNetworkExtractor
{

public:

	DeepDriveRoadNetworkExtractor(UWorld *world, SDeepDriveRoadNetwork &roadNetwork);

	void extract();

	uint32 getRoadLink(ADeepDriveRoadLinkProxy *linkProxy);

private:

	// add segment based on segment proxy
	uint32 addSegment(ADeepDriveRoadSegmentProxy &segmentProxy, const SDeepDriveRoadLink *link, EDeepDriveLaneType laneType);

	// add segment based on link proxy
	uint32 addSegment(ADeepDriveRoadLinkProxy &linkProxy, const SDeepDriveRoadLink *link);

	uint32 addStraightConnectionSegment(uint32 fromSegment, uint32 toSegment, float speedLimit, float slowDownDistance, bool generateCurve);

	uint32 addLink(ADeepDriveRoadLinkProxy &linkProxy);

	void addJunction();

	float calcHeading(const FVector &from, const FVector &to);

	FString buildSegmentName(const FString &linkName);

	UWorld							*m_World = 0;
	SDeepDriveRoadNetwork 			&m_RoadNetwork;

	TMap<FString, uint32>			m_SegmentCache;
	TMap<FString, uint32>			m_LinkCache;

	uint32							m_nextSegmentId = 1;
	uint32							m_nextLinkId = 1;
	uint32							m_nextJunctionId = 1;
};

inline FString DeepDriveRoadNetworkExtractor::buildSegmentName(const FString &linkName)
{
	return linkName + "_RS";
}
