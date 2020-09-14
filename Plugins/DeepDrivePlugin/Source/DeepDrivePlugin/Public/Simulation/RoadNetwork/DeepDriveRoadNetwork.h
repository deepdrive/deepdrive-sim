
#pragma once

#include "Engine.h"
#include "Runtime/CoreUObject/Public/UObject/ObjectMacros.h"

#include "Runtime/Engine/Classes/Components/SplineComponent.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveRoadNetwork, Log, All);

class ADeepDriveTrafficLight;
class ADeepDriveAgent;

UENUM(BlueprintType)
enum class EDeepDriveLaneType : uint8
{
	MAJOR_LANE			= 0	UMETA(DisplayName = "Major Lane"),
	ADDITIONAL_LANE 	= 1	UMETA(DisplayName = "Additional Lane"),
	PARKING_LANE	    = 2	UMETA(DisplayName = "Parking Lane"),
	CONNECTION	    	= 3	UMETA(DisplayName = "Connection")
};

UENUM(BlueprintType)
enum class EDeepDriveConnectionShape : uint8
{
	NO_CONNECTION 		= 0	UMETA(DisplayName = "No Connection"),
	STRAIGHT_LINE 		= 1	UMETA(DisplayName = "Straight Line"),
	QUADRATIC_SPLINE 	= 2	UMETA(DisplayName = "Quadratic Spline"),
	CUBIC_SPLINE    	= 3	UMETA(DisplayName = "Cubic Spline"),
	UTURN_SPLINE    	= 4	UMETA(DisplayName = "U-Turn Spline"),
	ROAD_SEGMENT		= 5 UMETA(DisplayName = "Road Segment")
};

UENUM(BlueprintType)
enum class EDeepDriveUTurnMode : uint8
{
	NOT_POSSIBLE 		= 0	UMETA(DisplayName = "Not Possible"),
	END_OF_LINK_ONLY	= 1	UMETA(DisplayName = "End of Link Only"),
	EVERYWHERE_ON_LINK	= 2	UMETA(DisplayName = "Everywhere on Link"),
	SEGMENTS	    	= 3	UMETA(DisplayName = "Segments")
};

UENUM(BlueprintType)
enum class EDeepDriveRoadPriority : uint8
{
	MAIN_ROAD			= 0	UMETA(DisplayName = "Main Road"),
	MINOR_ROAD			= 1	UMETA(DisplayName = "Minor Road")
};


UENUM(BlueprintType)
enum class EDeepDriveRightOfWay : uint8
{
	RIGHT_OF_WAY 		= 0	UMETA(DisplayName = "Right of Way"),
	YIELD	    		= 1	UMETA(DisplayName = "Yield"),
	STOP	    		= 2	UMETA(DisplayName = "Stop"),
	ALL_WAY_STOP   		= 3	UMETA(DisplayName = "All-Way Stop"),
	RIGHT_BEFORE_LEFT	= 4	UMETA(DisplayName = "RightBeforeLeft"),
	LEFT_BEFORE_RIGHT	= 5	UMETA(DisplayName = "LeftBeforeRight")
};

UENUM(BlueprintType)
enum class EDeepDriveManeuverType : uint8
{
	UNDEFINED	 			= 0 UMETA(DisplayName = "Undefined"),
	TURN_RIGHT				= 1 UMETA(DisplayName = "Turn Right"),
	GO_ON_STRAIGHT			= 2 UMETA(DisplayName = "Go on straight"),
	TURN_LEFT				= 3 UMETA(DisplayName = "Turn Left"),
	TRAFFIC_CIRCLE			= 4 UMETA(DisplayName = "Traffic Circle")
};

UENUM(BlueprintType)
enum class EDeepDriveJunctionType : uint8
{
	PASS_THROUGH 			= 0 UMETA(DisplayName = "Pass Through"),
	FOUR_WAY_JUNCTION		= 1 UMETA(DisplayName = "Four Way Junction"),
	T_JUNCTION				= 2 UMETA(DisplayName = "T-Junction"),
	TRAFFIC_CIRCLE			= 3 UMETA(DisplayName = "Traffic Circle"),

	DESTINATION_REACHED		= 255 UMETA(Hidden)
};

UENUM(BlueprintType)
enum class EDeepDriveJunctionSubType : uint8
{
	AUTO_DETECT			=  0 UMETA(DisplayName = "Undefined"),
	SUB_TYPE_1			=  1 UMETA(DisplayName = "Sub Type 1"),
	SUB_TYPE_2			=  2 UMETA(DisplayName = "Sub Type 2"),
	SUB_TYPE_3			=  3 UMETA(DisplayName = "Sub Type 3"),
	SUB_TYPE_4			=  4 UMETA(DisplayName = "Sub Type 4"),
	SUB_TYPE_5			=  5 UMETA(DisplayName = "Sub Type 5"),
	SUB_TYPE_6			=  6 UMETA(DisplayName = "Sub Type 6"),
	SUB_TYPE_7			=  7 UMETA(DisplayName = "Sub Type 7"),
	SUB_TYPE_8			=  8 UMETA(DisplayName = "Sub Type 8"),
	SUB_TYPE_9			=  9 UMETA(DisplayName = "Sub Type 9"),
	SUB_TYPE_10			= 10 UMETA(DisplayName = "Sub Type 10"),
	SUB_TYPE_11			= 11 UMETA(DisplayName = "Sub Type 11"),
	SUB_TYPE_12			= 12 UMETA(DisplayName = "Sub Type 12"),
	SUB_TYPE_13			= 13 UMETA(DisplayName = "Sub Type 13"),
	SUB_TYPE_14			= 14 UMETA(DisplayName = "Sub Type 14"),
	SUB_TYPE_15			= 15 UMETA(DisplayName = "Sub Type 15"),
	SUB_TYPE_16			= 16 UMETA(DisplayName = "Sub Type 16")
};


struct SDeepDriveRoadSegment
{
	uint32						SegmentId = 0;

	FVector						StartPoint = FVector::ZeroVector;
	FVector						EndPoint = FVector::ZeroVector;

	float						Heading = 0.0f;
	EDeepDriveLaneType			LaneType = EDeepDriveLaneType::MAJOR_LANE;

	FSplineCurves				SplineCurves;
	FTransform					SplineTransform;

	uint32						LinkId = 0;

	float						SpeedLimit = -1.0f;
	EDeepDriveConnectionShape	ConnectionShape = EDeepDriveConnectionShape::NO_CONNECTION;
	float						SlowDownDistance = -1.0f;				//	should be detected automatically
	float						CustomCurveParams[8];

	SDeepDriveRoadSegment		*LeftLane = 0;
	SDeepDriveRoadSegment		*RightLane = 0;

	FVector findClosestPoint(const FVector &pos) const;

	float getHeading(const FVector &pos) const;

	bool hasSpline() const
	{
		return SplineCurves.Position.Points.Num() > 0;
	}

	FVector getLocationOnSegment(float relativePos) const;

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

	FVector						StartDirection = FVector::ZeroVector;
	FVector						EndDirection = FVector::ZeroVector;

	FVector						StopLineLocation;

	EDeepDriveRoadPriority		RoadPriority;

	uint32						OppositeDirectionLink = 0;

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

struct SDeepDriveTurnDefinition
{
	uint32						FromLinkId = 0;
	uint32						ToLinkId = 0;
	EDeepDriveManeuverType		ManeuverType;
	FVector						WaitingLocation;

	ADeepDriveTrafficLight 		*TrafficLight = 0;
};

struct SDeepDriveJunctionConnection
{
	uint32					SegmentId = 0;
	uint32					ToSegmentId = 0;
	uint32					ConnectionSegmentId = 0;

};

struct SDeepDriveJunctionEntry
{
	uint32									LinkId;

	TArray<SDeepDriveTurnDefinition>		TurnDefinitions;

	EDeepDriveJunctionSubType				JunctionSubType;
	EDeepDriveRightOfWay					RightOfWay;
	FVector									ManeuverEntryPoint;
	FVector									LineOfSight;

	TArray<SDeepDriveJunctionConnection>	Connections;	

	ADeepDriveTrafficLight* getTrafficLight(uint32 toLinkId) const;

};

struct SDeepDriveJunction
{
	uint32									JunctionId = 0;
	FVector									Center = FVector::ZeroVector;

	EDeepDriveJunctionType					JunctionType;

	TArray<SDeepDriveJunctionEntry>			Entries;
	TArray<uint32>				 			LinksOut;

	bool findJunctionConnection(uint32 fromLinkId, uint32 fromSegment, uint32 toSegment, SDeepDriveJunctionConnection &junctionConnection) const;

	int32 findJunctionEntry(uint32 fromLinkId, const SDeepDriveJunctionEntry* &junctionEntry) const;

	bool isTurningAllowed(uint32 fromLinkId, uint32 toLinkId) const;

	EDeepDriveManeuverType getManeuverType(uint32 fromLinkId, uint32 toLinkId) const;

	void getRelevantAgents(uint32 fromLinkId, uint32 toLinkId, ADeepDriveAgent *egoAgent, TArray<ADeepDriveAgent *> &agents) const;

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

	TMap<uint32, FString>					LinkNameMap;

	uint32 findClosestLink(const FVector &pos) const;
	uint32 findClosestSegment(const FVector &pos, EDeepDriveLaneType laneType) const;
	FVector getLocationOnLink(uint32 linkId, EDeepDriveLaneType laneType, float t) const;

	FString getDebugLinkName(uint32 linkId) const;

};

namespace DeepDriveRoadNetwork
{
	const float SpeedLimitInTown = 50.0f;
	const float SpeedLimitConnection = 15.0f;
}
