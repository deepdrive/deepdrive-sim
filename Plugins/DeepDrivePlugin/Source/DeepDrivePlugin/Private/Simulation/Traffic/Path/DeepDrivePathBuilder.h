
#pragma once

#include "Simulation/Traffic/Path/DeepDrivePathDefines.h"

struct SDeepDriveRoadNetwork;
class UBezierCurveComponent;

class DeepDrivePathBuilder
{
public:

	DeepDrivePathBuilder(const SDeepDriveRoadNetwork &roadNetwork, TDeepDrivePathPoints &path, UBezierCurveComponent &bezierCmp);

	void buildPath(const TArray<SDeepDriveBasePathSegment> &baseSegments, TDeepDriveManeuvers &maneuvers);

	void buildPath(const SDeepDriveRoadSegment &from, const SDeepDriveRoadSegment &to, const SDeepDriveRoadSegment &connection, float fromLength, float toLength);

	FBox2D getArea() const;

private:

	float addRoadSegment(const SDeepDriveRoadSegment &segment,float carryOverDistance, float maxLength);
	float addStraightLine(uint32 segmentId, const FVector &start, const FVector &end, float carryOverDistance, float maxLength);
	float addConnectionSegment(const SDeepDriveRoadSegment &fromSegment, const SDeepDriveRoadSegment &toSegment, const SDeepDriveRoadSegment &connectionSegment, float carryOverDistance);
	float addQuadraticConnectionSegment(const SDeepDriveRoadSegment &fromSegment, const SDeepDriveRoadSegment &toSegment, const SDeepDriveRoadSegment &connectionSegment, float carryOverDistance);
	float addCubicConnectionSegment(const SDeepDriveRoadSegment &fromSegment, const SDeepDriveRoadSegment &toSegment, const SDeepDriveRoadSegment &connectionSegment, float carryOverDistance);
	float addUTurnConnectionSegment(const SDeepDriveRoadSegment &fromSegment, const SDeepDriveRoadSegment &toSegment, const SDeepDriveRoadSegment &connectionSegment, float carryOverDistance);

	void trimFromStart(float distance);
	void trimFromEnd(float distance);

	int32 rewind(float distance);

	void extractTangentFromSegment(const SDeepDriveRoadSegment &segment, FVector &start, FVector &end, bool atStart);


	const SDeepDriveRoadNetwork			&m_RoadNetwork;
	TDeepDrivePathPoints				&m_PathPoints;
	UBezierCurveComponent				&m_BezierCurve;

	const float							m_StepSize = 100.0f;

};
