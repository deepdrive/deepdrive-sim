
#pragma once

#include "Engine.h"
#include "Runtime/CoreUObject/Public/UObject/ObjectMacros.h"

#include "Runtime/Engine/Classes/Components/SplineComponent.h"

UENUM(BlueprintType)
enum class EDeepDriveLaneType : uint8
{
	MAJOR_LANE			= 0	UMETA(DisplayName = "Major Lane"),
	ADDITIONAL_LANE 	= 1	UMETA(DisplayName = "Additional Lane"),
	PARKING_LANE	    = 2	UMETA(DisplayName = "Parking Lane"),
	CONNECTION	    	= 3	UMETA(DisplayName = "Connection")
};


struct SDeepDriveRoadSegment
{
	uint32						SegmentId = 0;

	FVector						StartPoint = FVector::ZeroVector;
	FVector						EndPoint = FVector::ZeroVector;

	float						Heading = 0.0f;
	EDeepDriveLaneType			LaneType = EDeepDriveLaneType::MAJOR_LANE;

	TArray<FSplinePoint>		SplinePoints;
	FSplineCurves				SplineCurves;
	FTransform					Transform;

	uint32						LinkId = 0;

	float						SpeedLimit = -1.0f;
	bool						IsConnection = false;
	bool						GenerateCurve = false;
	float						SlowDownDistance = -1.0f;

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
	uint32						LinkId = 0;

	FVector						StartPoint = FVector::ZeroVector;
	FVector						EndPoint = FVector::ZeroVector;

	uint32						FromJunctionId = 0;
	uint32						ToJunctionId = 0;

	float						Heading = 0.0f;
	float						SpeedLimit = -1.0f;

	TArray<SDeepDriveLane>		Lanes;

	int32 getRightMostLane(EDeepDriveLaneType type) const;

};

struct SDeepDriveLaneConnection
{
	uint32		FromSegment = 0;
	uint32		ToSegment = 0;
	uint32		ConnectionSegment = 0;
};

struct SDeepDriveJunction
{
	uint32								JunctionId = 0;
	FVector								Center = FVector::ZeroVector;

	TArray<uint32> 						LinksIn;
	TArray<uint32>				 		LinksOut;

	TArray<SDeepDriveLaneConnection>	Connections;

	uint32 findConnectionSegment(uint32 fromSegment, uint32 toSegment) const;

};

struct SDeepDriveRouteData
{
	FVector				Start = FVector::ZeroVector;
	FVector				Destination = FVector::ZeroVector;

	TArray<uint32>		Links;

};

struct SDeepDriveRoadNetwork
{
	TMap<uint32, SDeepDriveJunction>		Junctions;
	TMap<uint32, SDeepDriveRoadLink>		Links;
	TMap<uint32, SDeepDriveRoadSegment> 	Segments;

	uint32 findClosestLink(const FVector &pos) const;
	uint32 findClosestSegment(const FVector &pos, EDeepDriveLaneType laneType) const;

};

namespace DeepDriveRoadNetwork
{
	const float SpeedLimitInTown = 50.0f;
	const float SpeedLimitConnection = 15.0f;
}
