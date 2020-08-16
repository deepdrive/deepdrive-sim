#pragma once

#include "CoreMinimal.h"
#include "Private/Simulation/Traffic/Path/DeepDrivePathDefines.h"

#include "Private/Simulation/Misc/MovingAverage.h"

#include "Private/Simulation/Misc/PIDController.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDrivePartialPath, Log, All);

struct SDeepDriveRoadNetwork;
class ADeepDriveAgent;
class UBezierCurveComponent;
struct SDeepDrivePathConfiguration;

class DeepDrivePartialPath
{

public:

	DeepDrivePartialPath(ADeepDriveAgent &agent, const SDeepDriveRoadNetwork &roadNetwork, UBezierCurveComponent &bezierCmp, const SDeepDrivePathConfiguration &pathCfg);

	void setup(const TArray<SDeepDriveBasePathSegment> &baseSegments);

	void trimStart(const FVector &startPos);
	void trimEnd(const FVector &endPos);

	void annotate();
	
	/*
		Return false when path is outdated, otherwise true
	*/
	bool update();
	void advance(float deltaSeconds, float &speed, float &steering, float &brake);

	float getRouteLength() const;

	bool hasReachedDestination() const;

	float getDistanceAlongRoute() const;

	float getDistanceToCenterOfTrack() const;

	void showPath(UWorld *world);

	int32 findClosestPathPoint(const FVector &location, float *t);

	int32 windForward(int32 fromIndex, float distanceCM, float *remainingDistance = 0) const;

	int32 rewind(int32 fromIndex, float distanceCM, float *remainingDistance = 0) const;

	float getDistance(int32 fromIndex, int32 toIndex);

	static float SmootherStep(float edge0, float edge1, float val);

private:

	void findClosestPathPoint(const FVector &location);

	float calcSteering(float dT);

	ADeepDriveAgent 					&m_Agent;
	const SDeepDriveRoadNetwork			&m_RoadNetwork;
	UBezierCurveComponent				&m_BezierCurve;

	const SDeepDrivePathConfiguration 	&m_PathConfiguration;

	TDeepDrivePathPoints 				m_PathPoints;
	int32								m_curPathPointIndex = -1;
	float								m_curPathSegmentT = 0.0f;
	float								m_Length = 0.0f;

	bool								m_hasReachedDestination = false;

	TDeepDriveManeuvers					m_Maneuvers;

	FVector								m_curAgentLocation;


	const float							m_StepSize = 100.0f;

	PIDController						m_SteeringPIDCtrl;

	TMovingAverage<5>					m_SteeringSmoother;

	float								m_totalTrackError = 0.0f;

};

inline bool DeepDrivePartialPath::hasReachedDestination() const
{
    return m_hasReachedDestination;
}
