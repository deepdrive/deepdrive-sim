
#pragma once

#include "CoreMinimal.h"

struct SDeepDriveRoadNetwork;
struct SDeepDriveRoadLink;

class ADeepDriveRoadLinkProxy;
class ADeepDriveRoadSegmentProxy;

class DeepDriveRoadNetworkExtractor
{

public:

	DeepDriveRoadNetworkExtractor(UWorld *world);

	void extract(SDeepDriveRoadNetwork &roadNetwork);

	uint32 getRoadLink(ADeepDriveRoadLinkProxy *linkProxy);

private:

	// add segment based on segment proxy
	uint32 addSegment(SDeepDriveRoadNetwork &roadNetwork, ADeepDriveRoadSegmentProxy &segmentProxy, const SDeepDriveRoadLink *link);

	// add segment based on link proxy
	uint32 addSegment(SDeepDriveRoadNetwork &roadNetwork, ADeepDriveRoadLinkProxy &linkProxy, const SDeepDriveRoadLink *link);

	uint32 addConnectionSegment(SDeepDriveRoadNetwork &roadNetwork, uint32 fromSegment, uint32 toSegment);

	uint32 addLink(SDeepDriveRoadNetwork &roadNetwork, ADeepDriveRoadLinkProxy &linkProxy);

	void addJunction();

	float calcHeading(const FVector &from, const FVector &to);

	FString buildSegmentName(const FString &linkName);

	UWorld							*m_World = 0;

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
