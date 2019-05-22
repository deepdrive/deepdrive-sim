

#include "DeepDrivePluginPrivatePCH.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoute.h"
#include "Runtime/Engine/Classes/Components/SplineComponent.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"
#include "Public/Simulation/Misc/BezierCurveComponent.h"


DEFINE_LOG_CATEGORY(LogDeepDriveRoute);

// Sets default values
ADeepDriveRoute::ADeepDriveRoute()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

	m_BezierCurve = CreateDefaultSubobject<UBezierCurveComponent>(TEXT("BezierCurve"));
}

void ADeepDriveRoute::Tick(float DeltaTime)
{
    return;
	if(m_RoutePoints.Num() > 0)
	{
		FColor col = FColor::Green;
		const uint8 prio = 40;

		DrawDebugPoint(GetWorld(), m_RoutePoints[0].Location, 10.0f, col, false, 0.0f, prio);
		for(signed i = 1; i < m_RoutePoints.Num(); ++i)
		{
			DrawDebugPoint(GetWorld(), m_RoutePoints[i].Location, 10.0f, col, false, 0.0f, prio);
			DrawDebugLine(GetWorld(), m_RoutePoints[i - 1].Location, m_RoutePoints[i].Location, col, false, 0.0f, prio, 4.0f);
		}
	}
}

void ADeepDriveRoute::initialize(const SDeepDriveRoadNetwork &roadNetwork, const SDeepDriveRouteData &routeData)
{
	m_RoadNetwork = &roadNetwork;
	m_RouteData = routeData;
}

void ADeepDriveRoute::convert(const FVector &location)
{
	convertToPoints(location);
	if (m_RoutePoints.Num() > 0)
	{
		trim(location, m_RouteData.Destination);
		annotateRoute();
	}
	UE_LOG(LogDeepDriveRoute, Log, TEXT("Route converted numPoints %d length %f"), m_RoutePoints.Num(), m_RouteLength);
}

void ADeepDriveRoute::trim(const FVector &start, const FVector &end)
{
	const int32 numRoutePoints = m_RoutePoints.Num();
	float bestDistStart = TNumericLimits<float>::Max();
	float bestDistEnd = TNumericLimits<float>::Max();
	int32 bestIndStart = -1;
	int32 bestIndEnd = -1;
	for(signed i = numRoutePoints - 1; i >= 0; --i)
	{
		const float curDistStart = (start - m_RoutePoints[i].Location).Size();
		if (curDistStart <= bestDistStart)
		{
			bestDistStart = curDistStart;
			bestIndStart = i;
		}
		const float curDistEnd = (end - m_RoutePoints[i].Location).Size();
		if (curDistEnd <= bestDistEnd)
		{
			bestDistEnd = curDistEnd;
			bestIndEnd = i;
		}
	}

	if(bestIndStart >= 0)
	{
		RoutePoints trimmedPoints;
		for(int32 i = bestIndStart; i < bestIndEnd; ++i)
			trimmedPoints.Add(m_RoutePoints[i]);

		m_RoutePoints = trimmedPoints;
		UE_LOG(LogDeepDriveRoute, Log, TEXT("Trimmed %d %s %s"), m_RoutePoints.Num(), *(m_RoutePoints[0].Location.ToString()), *(m_RoutePoints[m_RoutePoints.Num() - 1].Location.ToString()) );
	}
}

void ADeepDriveRoute::convertToPoints(const FVector &location)
{
	m_RouteLength = 0.0f;
	m_RoutePoints.Empty();
	if	(	m_RoadNetwork
		&&	m_RouteData.Links.Num() > 0
		)
	{
		float carryOverDistance = 0.0f;
		uint32 curLane = m_RoadNetwork->Links[m_RouteData.Links[0]].getRightMostLane(EDeepDriveLaneType::MAJOR_LANE);
		for (signed i = 0; i < m_RouteData.Links.Num(); ++i)
		{
			const SDeepDriveRoadLink &link = m_RoadNetwork->Links[m_RouteData.Links[i]];
			const bool lastLink = (i + 1) == m_RouteData.Links.Num();

			const SDeepDriveLane &lane = link.Lanes[curLane];

			for (signed j = 0; j < lane.Segments.Num(); ++j)
			{
				const SDeepDriveRoadSegment &segment = m_RoadNetwork->Segments[lane.Segments[j]];
				const bool lastSegment = (j + 1) == lane.Segments.Num();
				carryOverDistance = addSegmentToPoints(segment, lastLink && lastSegment, carryOverDistance);
			}

			if (!lastLink)
			{
				const SDeepDriveRoadLink &nextLink = m_RoadNetwork->Links[m_RouteData.Links[i + 1]];
				const SDeepDriveRoadSegment &segment = m_RoadNetwork->Segments[lane.Segments[lane.Segments.Num() - 1]];

				curLane = nextLink.getRightMostLane(EDeepDriveLaneType::MAJOR_LANE);

				const SDeepDriveJunction &junction = m_RoadNetwork->Junctions[link.ToJunctionId];
				const uint32 connectionSegmentId = junction.findConnectionSegment(segment.SegmentId, nextLink.Lanes[curLane].Segments[0]);
				if (connectionSegmentId)
				{
					const SDeepDriveRoadSegment &connectionSegment = m_RoadNetwork->Segments[connectionSegmentId];
					switch(connectionSegment.ConnectionShape)
					{
						case EDeepDriveConnectionShape::STRAIGHT_LINE:
							break;
						case EDeepDriveConnectionShape::QUADRATIC_SPLINE:
							carryOverDistance =	addQuadraticConnectionSegment(m_RoadNetwork->Segments[segment.SegmentId], m_RoadNetwork->Segments[nextLink.Lanes[curLane].Segments[0]], connectionSegment, carryOverDistance);
							break;
						case EDeepDriveConnectionShape::CUBIC_SPLINE:
							carryOverDistance = addCubicConnectionSegment(m_RoadNetwork->Segments[segment.SegmentId], m_RoadNetwork->Segments[nextLink.Lanes[curLane].Segments[0]], connectionSegment, carryOverDistance);
							break;
						case EDeepDriveConnectionShape::UTURN_SPLINE:
							carryOverDistance = addUTurnConnectionSegment(m_RoadNetwork->Segments[segment.SegmentId], m_RoadNetwork->Segments[nextLink.Lanes[curLane].Segments[0]], connectionSegment, carryOverDistance);
							break;
						case EDeepDriveConnectionShape::ROAD_SEGMENT:
							carryOverDistance =	addSegmentToPoints(connectionSegment, false, carryOverDistance);
							break;
					}
				}
			}
		}
	}
}

void ADeepDriveRoute::annotateRoute()
{
	const int32 numPoints = m_RoutePoints.Num();

	// calculate route length
	FVector last = m_RoutePoints[0].Location;
	m_RoutePoints[0].RemainingDistance = 0.0f;
	for (signed i = 1; i < numPoints; ++i)
	{
		const FVector &cur = m_RoutePoints[i].Location;
		m_RouteLength += (cur - last).Size();
		last = cur;
		m_RoutePoints[i].RemainingDistance = m_RouteLength;
	}

	// correct remaining distance and setup speed limits
	float speedLimit = DeepDriveRoadNetwork::SpeedLimitInTown;
	uint32 curSegmentId = 0;
	for (int32 i = 0; i < numPoints; ++i)
	{
		SRoutePoint &rp = m_RoutePoints[i];

		//	correct remaining distance
		rp.RemainingDistance = m_RouteLength - rp.RemainingDistance;

		if(rp.SegmentId != curSegmentId)
		{
			curSegmentId = rp.SegmentId;
			const SDeepDriveRoadSegment &segment = m_RoadNetwork->Segments[curSegmentId];
			speedLimit = segment.SpeedLimit;
			if(segment.ConnectionShape != EDeepDriveConnectionShape::NO_CONNECTION && segment.SlowDownDistance > 0.0f)
			{
				float coveredDist = 0.0f;
				FVector lastPoint = m_RoutePoints[i].Location;
				for(int32 j = i - 1; j >= 0; --j)
				{
					const FVector curPoint = m_RoutePoints[j].Location;
					coveredDist += (curPoint - lastPoint).Size();
					if (coveredDist < segment.SlowDownDistance)
					{
						m_RoutePoints[j].SpeedLimit = speedLimit;
						lastPoint = curPoint;
					}
					else
						break;
				}
			}
		}

		rp.SpeedLimit = speedLimit;
	}
}

float ADeepDriveRoute::addSegmentToPoints(const SDeepDriveRoadSegment &segment, bool addEnd, float carryOverDistance)
{
	float curDist = carryOverDistance;
	float segmentLength = 0.0f;
	if (segment.hasSpline())
	{
		segmentLength = segment.SplineCurves.GetSplineLength();
		while (curDist < segmentLength)
		{
			SRoutePoint rp;
			rp.SegmentId = segment.SegmentId;
			rp.RelativePosition = curDist / segmentLength;
			const float key = segment.SplineCurves.ReparamTable.Eval(curDist, 0.0f);
			rp.Location = segment.SplineTransform.TransformPosition(segment.SplineCurves.Position.Eval(key, FVector::ZeroVector));

			m_RoutePoints.Add(rp);

			curDist += m_StepSize;
		}
	}
	else
	{
		const FVector dir = segment.EndPoint - segment.StartPoint;
		segmentLength = dir.Size();
		while (curDist < segmentLength)
		{
			SRoutePoint rp;
			rp.SegmentId = segment.SegmentId;
			rp.RelativePosition = curDist / segmentLength;
			rp.Location = rp.RelativePosition * dir + segment.StartPoint;
			m_RoutePoints.Add(rp);

			curDist += m_StepSize;
		}
	}
	carryOverDistance = curDist - segmentLength;
	return carryOverDistance;
}

float ADeepDriveRoute::addQuadraticConnectionSegment(const SDeepDriveRoadSegment &fromSegment, const SDeepDriveRoadSegment &toSegment, const SDeepDriveRoadSegment &connectionSegment, float carryOverDistance)
{
	FVector dir = fromSegment.EndPoint - fromSegment.StartPoint;
	FVector nrm(-dir.Y, dir.X, 0.0f);
	dir.Normalize();
	nrm.Normalize();

	const FVector &p0 = fromSegment.EndPoint;
	const FVector &p2 = toSegment.StartPoint;
	FVector p1 = p0 + dir * connectionSegment.CustomCurveParams[0] + nrm * connectionSegment.CustomCurveParams[1];

	for(float t = 0.0f; t < 1.0f; t+= 0.05f)
	{
		SRoutePoint rp;
		rp.SegmentId = connectionSegment.SegmentId;
		rp.RelativePosition = t;

		const float oneMinusT = (1.0f - t);
		rp.Location = oneMinusT * oneMinusT * p0 + 2.0f * oneMinusT * t * p1 + t * t *p2;

		m_RoutePoints.Add(rp);
	}

	return 0.0f;
}

float ADeepDriveRoute::addCubicConnectionSegment(const SDeepDriveRoadSegment &fromSegment, const SDeepDriveRoadSegment &toSegment, const SDeepDriveRoadSegment &connectionSegment, float carryOverDistance)
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

	for(float t = 0.0f; t < 1.0f; t+= 0.05f)
	{
		SRoutePoint rp;
		rp.SegmentId = connectionSegment.SegmentId;
		rp.RelativePosition = t;

		const float oneMinusT = (1.0f - t);
		rp.Location = oneMinusT * oneMinusT * oneMinusT * p0 + (3.0f * oneMinusT * oneMinusT * t) * p1 + (3.0f * oneMinusT * t * t) * p2 + t * t * t * p3;

		m_RoutePoints.Add(rp);
	}

	return 0.0f;
}

float ADeepDriveRoute::addUTurnConnectionSegment(const SDeepDriveRoadSegment &fromSegment, const SDeepDriveRoadSegment &toSegment, const SDeepDriveRoadSegment &connectionSegment, float carryOverDistance)
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

	UE_LOG(LogDeepDriveRoute, Log, TEXT("addUTurnConnectionSegment dir %s nrm %s"), *(dir.ToString()), *(nrm.ToString()) );


	FVector middle = 0.5f * (fromEnd + toStart);

	m_BezierCurve->ClearControlPoints();
	m_BezierCurve->AddControlPoint(fromEnd);

	const float *params = connectionSegment.CustomCurveParams;
	m_BezierCurve->AddControlPoint(middle + nrm * params[3] - dir * params[4]);

	m_BezierCurve->AddControlPoint(middle + nrm * params[1] - dir * params[2]);
	m_BezierCurve->AddControlPoint(middle + nrm * params[0]);
	m_BezierCurve->AddControlPoint(middle + nrm * params[1] + dir * params[2]);

	m_BezierCurve->AddControlPoint(middle + nrm * params[3] + dir * params[4]);

	m_BezierCurve->AddControlPoint(toStart);

	for(float t = 0.0f; t < 1.0f; t+= 0.05f)
	{
		SRoutePoint rp;
		rp.SegmentId = connectionSegment.SegmentId;
		rp.RelativePosition = t;

		rp.Location = m_BezierCurve->Evaluate(t);

		m_RoutePoints.Add(rp);
	}

	return 0.0f;
}

void ADeepDriveRoute::update(ADeepDriveAgent &agent)
{
	m_curRoutePointIndex = findClosestRoutePoint(agent.GetActorLocation());
}

void ADeepDriveRoute::placeAgentAtStart(ADeepDriveAgent &agent)
{
	int32 index = findClosestRoutePoint(m_RouteData.Start);
	if(index >= 0)
	{
		const float heading = m_RoadNetwork->Segments[m_RoutePoints[index].SegmentId].Heading;
		FTransform transform(FRotator(0.0f, heading, 0.0f), m_RoutePoints[index].Location, FVector(1.0f, 1.0f, 1.0f));
		agent.SetActorTransform(transform, false, 0, ETeleportType::TeleportPhysics);
	}
}

float ADeepDriveRoute::getRemainingDistance()
{
	return		m_curRoutePointIndex >= 0 && m_curRoutePointIndex < m_RoutePoints.Num()
			?	m_RoutePoints[m_curRoutePointIndex].RemainingDistance
			: 	0.0f;
}

FVector ADeepDriveRoute::getLocationAhead(float distanceAhead, float sideOffset)
{
	int32 ind = m_curRoutePointIndex + static_cast<int32> (distanceAhead / m_StepSize);
	FVector locAhead = m_RoutePoints[ind % m_RoutePoints.Num()].Location;
	return locAhead;
}

float ADeepDriveRoute::getSpeedLimit(float distanceAhead)
{
	float speedLimit = 0.0f;
	if(m_curRoutePointIndex >= 0)
	{
		if(distanceAhead > 0.0f)
		{
			const int32 index = getPointIndexAhead(distanceAhead);
			if(index >= 0)
				speedLimit = m_RoutePoints[index].SpeedLimit;
		}
		else
			speedLimit = m_RoutePoints[m_curRoutePointIndex % m_RoutePoints.Num()].SpeedLimit;
	}

	return speedLimit;
}

int32 ADeepDriveRoute::findClosestRoutePoint(const FVector &location) const
{
	float bestDist = TNumericLimits<float>::Max();
	int32 index = -1;
	for(signed i = m_RoutePoints.Num() - 1; i >= 0; --i)
	{
		const float curDist = (location - m_RoutePoints[i].Location).Size();
		if (curDist <= bestDist)
		{
			bestDist = curDist;
			index = i;
		}
	}
	return index;
}

int32 ADeepDriveRoute::getPointIndexAhead(float distanceAhead) const
{
	int32 curIndex = m_curRoutePointIndex;
	FVector lastPoint = m_RoutePoints[curIndex].Location;
	for( ; curIndex < m_RoutePoints.Num() && distanceAhead > 0.0f; ++curIndex)
	{
		FVector curPoint = m_RoutePoints[curIndex].Location;
		distanceAhead -= (curPoint - lastPoint).Size();
		lastPoint = curPoint;
	}

	return curIndex < m_RoutePoints.Num() ? curIndex : m_RoutePoints.Num() - 1;
}

FVector ADeepDriveRoute::calcControlPoint(const SDeepDriveRoadSegment &segA, const SDeepDriveRoadSegment &segB)
{
	FVector aStart;
	FVector aEnd;
	FVector bStart;
	FVector bEnd;

	extractTangentFromSegment(segA, aStart, aEnd, false);
	extractTangentFromSegment(segB, bStart, bEnd, true);

	FVector2D r = FVector2D(aEnd - aStart);
	FVector2D s = FVector2D(bStart - bEnd);

	float dot = FMath::Abs(FVector2D::DotProduct(r, s) / (r.Size() * s.Size()));

	//r.Normalize();
	//s.Normalize();

	float cRS = FVector2D::CrossProduct(r, s);

	if (FMath::Abs(cRS) > 0.001f && dot < 0.975f)
	{
		FVector2D qp(bEnd - aStart);
		//qp.Normalize();
		float t = FVector2D::CrossProduct(qp, s) / cRS;
		FVector2D intersection(FVector2D(aStart) + r * t);
		const float z = 0.5f * (aEnd.Z + bStart.Z);

		return FVector(intersection, z);
	}


	UE_LOG(LogDeepDriveRoute, Log, TEXT("no intersection, taking center"));
	return 0.5f * (segA.EndPoint + segB.StartPoint); 
}

void ADeepDriveRoute::extractTangentFromSegment(const SDeepDriveRoadSegment &segment, FVector &start, FVector &end, bool atStart)
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
