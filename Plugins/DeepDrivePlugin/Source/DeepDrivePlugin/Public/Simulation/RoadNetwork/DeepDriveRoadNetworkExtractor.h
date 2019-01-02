
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

	uint32 addSegment(SDeepDriveRoadNetwork &roadNetwork, ADeepDriveRoadSegmentProxy &segmentProxy, const SDeepDriveRoadLink *link);

	uint32 addLink(SDeepDriveRoadNetwork &roadNetwork, ADeepDriveRoadLinkProxy &linkProxy);

	void addJunction();

	float calcHeading(const FVector &from, const FVector &to);

	UWorld							*m_World = 0;

	TMap<FString, uint32>			m_SegmentCache;
	TMap<FString, uint32>			m_LinkCache;

	uint32							m_nextSegmentId = 1;
	uint32							m_nextLinkId = 1;
	uint32							m_nextJunctionId = 1;
};
