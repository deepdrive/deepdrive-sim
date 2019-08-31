
#include "Simulation/Traffic/Path/DeepDrivePathBuilder.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoadNetwork.h"
#include "Public/Simulation/Misc/BezierCurveComponent.h"
#include "Private/Utils/DeepDriveUtils.h"

DeepDrivePathBuilder::DeepDrivePathBuilder(const SDeepDriveRoadNetwork &roadNetwork, TDeepDrivePathPoints &path, UBezierCurveComponent &bezierCmp)
	:	m_RoadNetwork(roadNetwork)
	,	m_PathPoints(path)
	,	m_BezierCurve(bezierCmp)
{
}

void DeepDrivePathBuilder::buildPath(const TArray<SDeepDriveBasePathSegment> &baseSegments)
{
	float carryOverDistance = 0.0f;
	for(const SDeepDriveBasePathSegment &baseSegment : baseSegments)
	{
		for(uint32 segId : baseSegment.Segments)
		{
			const SDeepDriveRoadSegment &roadSegment = m_RoadNetwork.Segments[segId];
			addRoadSegment(roadSegment, carryOverDistance, -1.0f);
		}

		if(baseSegment.Connection.SegmentId)
		{
			const SDeepDriveRoadSegment &fromSegment = m_RoadNetwork.Segments[baseSegment.Connection.SegmentId];
			const SDeepDriveRoadSegment &toSegment = m_RoadNetwork.Segments[baseSegment.Connection.ToSegmentId];
			const SDeepDriveRoadSegment &connectionSegment = m_RoadNetwork.Segments[baseSegment.Connection.ConnectionSegmentId];
			carryOverDistance = addConnectionSegment(fromSegment, toSegment, connectionSegment, carryOverDistance);
		}
	}
}

void DeepDrivePathBuilder::buildPath(const SDeepDriveRoadSegment &from, const SDeepDriveRoadSegment &to, const SDeepDriveRoadSegment &connection, float fromLength, float toLength)
{
	float carryOverDistance = addRoadSegment(from, 0.0f, -1.0f);
	trimFromStart(fromLength);
	carryOverDistance = addConnectionSegment(from, to, connection, carryOverDistance);
	addRoadSegment(to, carryOverDistance, toLength);
}

float DeepDrivePathBuilder::addRoadSegment(const SDeepDriveRoadSegment &segment, float carryOverDistance, float maxLength)
{
	if (segment.hasSpline())
	{
		float curDist = carryOverDistance;
		float segmentLength = 0.0f;
		segmentLength = segment.SplineCurves.GetSplineLength();
		while	(	curDist < segmentLength
				&&	(maxLength < 0.0f || curDist < maxLength)
				)
		{
			SDeepDrivePathPoint pathPoint;
			pathPoint.SegmentId = segment.SegmentId;
			const float key = segment.SplineCurves.ReparamTable.Eval(curDist, 0.0f);
			pathPoint.Location = segment.SplineTransform.TransformPosition(segment.SplineCurves.Position.Eval(key, FVector::ZeroVector));

			m_PathPoints.Add(pathPoint);

			curDist += m_StepSize;
		}
		carryOverDistance = curDist - segmentLength;
	}
	else
	{
		carryOverDistance = addStraightLine(segment.SegmentId, segment.StartPoint, segment.EndPoint, carryOverDistance, maxLength);
	}
	return carryOverDistance;
}

float DeepDrivePathBuilder::addStraightLine(uint32 segmentId, const FVector &start, const FVector &end, float carryOverDistance, float maxLength)
{
	float curDist = carryOverDistance;
	FVector dir(end - start);
	float segmentLength = dir.Size();
	while	(	curDist < segmentLength
			&&	(maxLength < 0.0f || curDist < maxLength)
			)
	{
		SDeepDrivePathPoint pathPoint;
		pathPoint.SegmentId = segmentId;
		const float relativePosition = curDist / segmentLength;
		pathPoint.Location = relativePosition * dir + start;
		m_PathPoints.Add(pathPoint);

		curDist += m_StepSize;
	}

	return curDist - segmentLength;
}


float DeepDrivePathBuilder::addConnectionSegment(const SDeepDriveRoadSegment &fromSegment, const SDeepDriveRoadSegment &toSegment, const SDeepDriveRoadSegment &connectionSegment, float carryOverDistance)
{
	switch(connectionSegment.ConnectionShape)
	{
		case EDeepDriveConnectionShape::STRAIGHT_LINE:
		carryOverDistance = addStraightLine(connectionSegment.SegmentId, fromSegment.EndPoint, toSegment.StartPoint, carryOverDistance, -1.0f);
			break;
		case EDeepDriveConnectionShape::QUADRATIC_SPLINE:
			carryOverDistance =	addQuadraticConnectionSegment(fromSegment, toSegment, connectionSegment, carryOverDistance);
			break;
		case EDeepDriveConnectionShape::CUBIC_SPLINE:
			carryOverDistance = addCubicConnectionSegment(fromSegment, toSegment, connectionSegment, carryOverDistance);
			break;
		case EDeepDriveConnectionShape::UTURN_SPLINE:
			carryOverDistance = addUTurnConnectionSegment(fromSegment, toSegment, connectionSegment, carryOverDistance);
			break;
		case EDeepDriveConnectionShape::ROAD_SEGMENT:
			// carryOverDistance =	addSegmentToPoints(connectionSegment, false, carryOverDistance);
			carryOverDistance =	addRoadSegment(connectionSegment, carryOverDistance, -1.0f);
			break;
	}

	return carryOverDistance;
}

float DeepDrivePathBuilder::addQuadraticConnectionSegment(const SDeepDriveRoadSegment &fromSegment, const SDeepDriveRoadSegment &toSegment, const SDeepDriveRoadSegment &connectionSegment, float carryOverDistance)
{
	FVector dir = fromSegment.EndPoint - fromSegment.StartPoint;
	FVector nrm(-dir.Y, dir.X, 0.0f);
	dir.Normalize();
	nrm.Normalize();

	const FVector &p0 = fromSegment.EndPoint;
	const FVector &p2 = toSegment.StartPoint;
	FVector p1 = p0 + dir * connectionSegment.CustomCurveParams[0] + nrm * connectionSegment.CustomCurveParams[1];

	for (float t = 0.0f; t < 1.0f; t += 0.05f)
	{
		SDeepDrivePathPoint pathPoint;
		pathPoint.SegmentId = connectionSegment.SegmentId;

		const float oneMinusT = (1.0f - t);
		pathPoint.Location = oneMinusT * oneMinusT * p0 + 2.0f * oneMinusT * t * p1 + t * t * p2;

		m_PathPoints.Add(pathPoint);
	}

	return 0.0f;
}

float DeepDrivePathBuilder::addCubicConnectionSegment(const SDeepDriveRoadSegment &fromSegment, const SDeepDriveRoadSegment &toSegment, const SDeepDriveRoadSegment &connectionSegment, float carryOverDistance)
{
	FVector fromStart;
	FVector fromEnd;
	FVector toStart;
	FVector toEnd;

	extractTangentFromSegment(fromSegment, fromStart, fromEnd, false);
	extractTangentFromSegment(toSegment, toStart, toEnd, true);

	FVector dir0 = fromEnd - fromStart;
	FVector dir1 = toStart - toEnd;
	dir0.Normalize();
	dir1.Normalize();

	const FVector &p0 = fromEnd;
	const FVector &p3 = toStart;
	FVector p1 = p0 + dir0 * connectionSegment.CustomCurveParams[0];
	FVector p2 = p3 + dir1 * connectionSegment.CustomCurveParams[1];

	for (float t = 0.0f; t < 1.0f; t += 0.05f)
	{
		SDeepDrivePathPoint pathPoint;
		pathPoint.SegmentId = connectionSegment.SegmentId;

		const float oneMinusT = (1.0f - t);
		pathPoint.Location = oneMinusT * oneMinusT * oneMinusT * p0 + (3.0f * oneMinusT * oneMinusT * t) * p1 + (3.0f * oneMinusT * t * t) * p2 + t * t * t * p3;

		m_PathPoints.Add(pathPoint);
	}

	return 0.0f;
}

float DeepDrivePathBuilder::addUTurnConnectionSegment(const SDeepDriveRoadSegment &fromSegment, const SDeepDriveRoadSegment &toSegment, const SDeepDriveRoadSegment &connectionSegment, float carryOverDistance)
{
	FVector fromStart;
	FVector fromEnd;
	FVector toStart;
	FVector toEnd;

	extractTangentFromSegment(fromSegment, fromStart, fromEnd, false);
	extractTangentFromSegment(toSegment, toStart, toEnd, true);

	FVector dir = toStart - fromEnd;
	dir.Normalize();
	FVector nrm(-dir.Y, dir.X, 0.0f);
	nrm.Normalize();

	FVector middle = 0.5f * (fromEnd + toStart);

	m_BezierCurve.ClearControlPoints();
	m_BezierCurve.AddControlPoint(fromEnd);

	const float *params = connectionSegment.CustomCurveParams;
	m_BezierCurve.AddControlPoint(middle + nrm * params[3] - dir * params[4]);

	m_BezierCurve.AddControlPoint(middle + nrm * params[1] - dir * params[2]);
	m_BezierCurve.AddControlPoint(middle + nrm * params[0]);
	m_BezierCurve.AddControlPoint(middle + nrm * params[1] + dir * params[2]);

	m_BezierCurve.AddControlPoint(middle + nrm * params[3] + dir * params[4]);

	m_BezierCurve.AddControlPoint(toStart);

	for(float t = 0.0f; t < 1.0f; t+= 0.05f)
	{
		SDeepDrivePathPoint pathPoint;
		pathPoint.SegmentId = connectionSegment.SegmentId;
		
		pathPoint.Location = m_BezierCurve.Evaluate(t);

		m_PathPoints.Add(pathPoint);
	}

	return 0.0f;
}

void DeepDrivePathBuilder::trimFromStart(float distance)
{
	int32 i = m_PathPoints.Num() - 1;
	for( ; i > 0 && distance > 0.0f; --i)
	{
		const float curLength = (m_PathPoints[i].Location - m_PathPoints[i - 1].Location).Size();
		distance -= curLength;
	}

	if(i > 0)
		m_PathPoints.RemoveAt(0, i);
}

void DeepDrivePathBuilder::trimFromEnd(float distance)
{
	int32 i = 1;
	for ( ; i < m_PathPoints.Num() && distance > 0.0f; ++i)
	{
		const float curLength = (m_PathPoints[i].Location - m_PathPoints[i - 1].Location).Size();
		distance -= curLength;
	}

	if (i < m_PathPoints.Num())
		m_PathPoints.RemoveAt(i, m_PathPoints.Num() - i);
}

FBox2D DeepDrivePathBuilder::getArea() const
{
	if(m_PathPoints.Num() > 0)
	{
		FBox2D area(FVector2D(TNumericLimits<float>::Max(), TNumericLimits<float>::Max()), FVector2D(TNumericLimits<float>::Min(), TNumericLimits<float>::Min()));

		for(auto &p : m_PathPoints)
		{
			deepdrive::utils::expandBox2D(area, FVector2D(p.Location));
		}

		return area;
	}
	return FBox2D(EForceInit::ForceInit);
}


void DeepDrivePathBuilder::extractTangentFromSegment(const SDeepDriveRoadSegment &segment, FVector &start, FVector &end, bool atStart)
{
	if (segment.hasSpline())
	{
		const int32 index = atStart ? 0 : (segment.SplineCurves.Position.Points.Num() - 1);
		const FInterpCurvePointVector& point = segment.SplineCurves.Position.Points[index];
		FVector direction = segment.SplineTransform.TransformVector(atStart ? point.ArriveTangent : point.LeaveTangent);
		direction.Normalize();
		direction *= 100.0f;
		if(atStart)
		{
			start = segment.StartPoint;
			end = segment.EndPoint + direction;
		}
		else
		{
			start = segment.EndPoint - direction;
			end = segment.EndPoint;
		}
	}
	else
	{
		start = segment.StartPoint;
		end = segment.EndPoint;
	}
}
