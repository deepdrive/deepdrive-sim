
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
	FVector						StartPoint;
	FVector						EndPoint;

	FSplineCurves				*Spline = 0;

	SDeepDriveRoadSegment		*LeftLane = 0;
	SDeepDriveRoadSegment		*RightLane = 0;

};


struct SDeepDriveLane
{
	EDeepDriveLaneType				LaneType = EDeepDriveLaneType::MAJOR_LANE;

	TArray<SDeepDriveRoadSegment>	Segments;
};

struct SDeepDriveRoadLink
{
	FVector						StartPoint;
	FVector						EndPoint;

	TArray<SDeepDriveLane>		Lanes;
};

struct SDeepDriveLaneConnection
{
	SDeepDriveRoadSegment	*FromSegment = 0;
	SDeepDriveRoadSegment	*ToSegment = 0;
	SDeepDriveRoadSegment	*ConnectionSegment = 0;
};

struct SDeepDriveJunction
{
	TArray<SDeepDriveRoadLink*>			RoadLinks;
	TArray<SDeepDriveLaneConnection>	Connections;
};

struct SDeepDriveRoute
{
	FVector							Start;
	FVector							Destination;

	TArray<SDeepDriveRoadLink*>		Links;

};

struct SDeepDriveRoadNetwork
{
	TArray<SDeepDriveJunction>		Junctions;
	TArray<SDeepDriveRoadLink>		Links;


};
