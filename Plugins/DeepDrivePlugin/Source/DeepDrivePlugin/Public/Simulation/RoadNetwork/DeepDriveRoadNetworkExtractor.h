
#pragma once

#include "CoreMinimal.h"

#include "Simulation/RoadNetwork/DeepDriveRoadNetwork.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveRoadNetworkExtractor, Log, All);

class ADeepDriveRoadLinkProxy;
class ADeepDriveRoadSegmentProxy;
struct FDeepDriveJunctionConnectionProxy;

class DeepDriveRoadNetworkExtractor
{

public:

	DeepDriveRoadNetworkExtractor(UWorld *world, SDeepDriveRoadNetwork &roadNetwork);

	void extract();

	uint32 getRoadLink(ADeepDriveRoadLinkProxy *linkProxy);

private:

	// add segment based on segment proxy
	uint32 addSegment(ADeepDriveRoadSegmentProxy &segmentProxy, const SDeepDriveRoadLink &link, EDeepDriveLaneType laneType);

	// add segment based on link proxy
	uint32 addSegment(ADeepDriveRoadLinkProxy &linkProxy, const SDeepDriveRoadLink &link);

	uint32 addConnectionSegment(ADeepDriveRoadSegmentProxy &segmentProxy);

	uint32 addConnectionSegment(uint32 fromSegment, uint32 toSegment, const FDeepDriveJunctionConnectionProxy &junctionConnectionProxy);

	uint32 addStraightConnectionSegment(uint32 fromSegment, uint32 toSegment, float speedLimit, bool generateCurve);

	uint32 addLink(ADeepDriveRoadLinkProxy &linkProxy);

	void addJunction();

	float calcHeading(const FVector &from, const FVector &to);

	FString buildSegmentName(const FString &linkName);

	float getSpeedLimit(float speedLimit);
	float getConnectionSpeedLimit(float speedLimit, EDeepDriveConnectionShape connectionShape);

	UWorld								*m_World = 0;
	SDeepDriveRoadNetwork 				&m_RoadNetwork;

	TMap<FString, uint32>				m_SegmentCache;
	TMap<FString, uint32>				m_LinkCache;
	TArray<ADeepDriveRoadLinkProxy*>	m_LinkProxies;

	uint32								m_nextSegmentId = 1;
	uint32								m_nextLinkId = 1;
	uint32								m_nextJunctionId = 1;
};

inline FString DeepDriveRoadNetworkExtractor::buildSegmentName(const FString &linkName)
{
	return linkName + "_RS";
}

inline float DeepDriveRoadNetworkExtractor::getSpeedLimit(float speedLimit)
{
	return speedLimit > 0.0f ? speedLimit : DeepDriveRoadNetwork::SpeedLimitInTown;
}

inline float DeepDriveRoadNetworkExtractor::getConnectionSpeedLimit(float speedLimit, EDeepDriveConnectionShape connectionShape)
{
	return		speedLimit > 0.0f
			?	speedLimit
			:	(connectionShape == EDeepDriveConnectionShape::STRAIGHT_LINE ? DeepDriveRoadNetwork::SpeedLimitInTown : DeepDriveRoadNetwork::SpeedLimitConnection);
}
