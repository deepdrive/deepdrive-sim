
#pragma once

#include "Engine.h"
#include "Runtime/CoreUObject/Public/UObject/ObjectMacros.h"

#include "Runtime/Engine/Classes/Components/SplineComponent.h"

UENUM(BlueprintType)
enum class EDeepDriveLaneType : uint8
{
	MAJOR_LANE			= 0	UMETA(DisplayName = "Major Lane"),
	ADDITIONAL_LANE 	= 1	UMETA(DisplayName = "Additional Lane"),
	PARKING_LANE	    = 2	UMETA(DisplayName = "Parking Lane")
};


struct SDeepDriveRoadSegment
{
	uint32						SegmentId;

	FVector						StartPoint;
	FVector						EndPoint;

	float						Heading;

	TArray<FSplinePoint>		SplinePoints;
	FSplineCurves				SplineCurves;
	FTransform					Transform;

	uint32						LinkId;

	TArray<FVector2D>			SpeedLimits;	// rel. position, speed limit

	SDeepDriveRoadSegment		*LeftLane = 0;
	SDeepDriveRoadSegment		*RightLane = 0;
};


struct SDeepDriveLane
{
	EDeepDriveLaneType		LaneType = EDeepDriveLaneType::MAJOR_LANE;

	TArray<uint32>			Segments;
};

struct SDeepDriveRoadLink
{
	uint32						LinkId;

	FVector						StartPoint;
	FVector						EndPoint;

	uint32						FromJunctionId;
	uint32						ToJunctionId;

	float						Heading;
	float						SpeedLimit;

	TArray<SDeepDriveLane>		Lanes;

};

struct SDeepDriveLaneConnection
{
	uint32		FromSegment = 0;
	uint32		ToSegment = 0;
	uint32		ConnectionSegment = 0;
};

struct SDeepDriveJunction
{
	uint32								JunctionId;

	TArray<uint32> 						LinksIn;
	TArray<uint32>				 		LinksOut;

	TArray<SDeepDriveLaneConnection>	Connections;

	uint32 findConnectionSegment(uint32 fromSegment, uint32 toSegment) const;

};

struct SDeepDriveRouteData
{
	FVector				Start;
	FVector				Destination;

	TArray<uint32>		Links;

};

struct SDeepDriveRoadNetwork
{
	TMap<uint32, SDeepDriveJunction>		Junctions;
	TMap<uint32, SDeepDriveRoadLink>		Links;
	TMap<uint32, SDeepDriveRoadSegment> 	Segments;

	float getSpeedLimit(uint32 segmentId, float relativePos) const;

};

namespace DeepDriveRoadNetwork
{
	const float SpeedLimitInTown = 50.0f;
	const float SpeedLimitConnection = 15.0f;
}
