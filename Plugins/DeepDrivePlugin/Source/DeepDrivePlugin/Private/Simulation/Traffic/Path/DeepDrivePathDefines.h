
#pragma once

#include "Public/Simulation/RoadNetwork/DeepDriveRoadNetwork.h"

class DeepDriveTrafficBehaviorTree;

struct SDeepDrivePathConfiguration
{
	FVector             PIDSteering;

	float               SpeedDownRange;
	float               SpeedUpRange;

};

struct SDeepDrivePathPoint
{
	FVector				Location;
	FVector2D			Direction;
	FVector2D			Normal;

	uint32				SegmentId;

	float               CurveRadius;
	float               CurveAngle;
	float               Heading;

	float				SpeedLimit;

	float				Distance;
	float				RemainingDistance;
};

typedef TArray<SDeepDrivePathPoint>	TDeepDrivePathPoints;

struct SDeepDriveCrossTrafficRoad
{
	SDeepDriveCrossTrafficRoad(EDeepDriveManeuverType maneuverType, ADeepDriveTrafficLight *trafficLight, uint32 from, uint32 to, float fromLength, float toLength)
		:	ManeuverType(maneuverType)
		,	TrafficLight(trafficLight)
		,	FromLinkId(from)
		,	ToLinkId(to)
		,	FromLength(fromLength)
		,	ToLength(toLength)
	{
	}

	EDeepDriveManeuverType				ManeuverType;
	ADeepDriveTrafficLight				*TrafficLight = 0;

	uint32								FromLinkId = 0;
	uint32								ToLinkId = 0;

	float								FromLength = 0.0f;
	float								ToLength = 0.0f;

	TArray<TDeepDrivePathPoints>		Paths;
};

typedef TArray<SDeepDriveCrossTrafficRoad> TDeepDriveCrossTrafficRoads;

struct SDeepDriveManeuver
{
	uint32									FromLinkId = 0;
	uint32									ToLinkId = 0;

	EDeepDriveJunctionType					JunctionType = EDeepDriveJunctionType::PASS_THROUGH;
	EDeepDriveJunctionSubType				JunctionSubType = EDeepDriveJunctionSubType::AUTO_DETECT;
	EDeepDriveManeuverType					ManeuverType = EDeepDriveManeuverType::UNDEFINED;
	EDeepDriveRightOfWay					RightOfWay = EDeepDriveRightOfWay::RIGHT_OF_WAY;

	EDeepDriveRoadPriority					FromRoadPriority = EDeepDriveRoadPriority::MINOR_ROAD;
	EDeepDriveRoadPriority					ToRoadPriority = EDeepDriveRoadPriority::MINOR_ROAD;

	FVector									EntryPoint;
	FVector									ExitPoint;

	FBox2D									ManeuverArea;

	ADeepDriveTrafficLight 					*TrafficLight = 0;

	DeepDriveTrafficBehaviorTree			*BehaviorTree = 0;

	// list of cross traffic relevant roads to check for traffic
	TDeepDriveCrossTrafficRoads				CrossTrafficRoads;

	int32									EntryPointIndex = -1;
	int32									ExitPointIndex = -1;

	int32									DirectionIndicationBeginIndex = -1;
	int32									DirectionIndicationEndIndex = -1;

};

typedef TArray<SDeepDriveManeuver>	TDeepDriveManeuvers;

struct SDeepDriveBasePathSegment
{
	TArray<uint32>						Segments;
	SDeepDriveJunctionConnection		Connection;

	SDeepDriveManeuver					Maneuver;
};


struct SDeepDriveRoute
{
	FVector						Start = FVector::ZeroVector;
	FVector						Destination = FVector::ZeroVector;

	TArray<uint32>				Links;
	TArray<SDeepDriveManeuver>	Maneuvers;		//	number of maneuvers should be one less than number of links
};

