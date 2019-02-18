
#include "DeepDrivePluginPrivatePCH.h"
#include "DeepDriveJunctionProxy.h"

#include "Public/Simulation/RoadNetwork/DeepDriveRoadLinkProxy.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoadSegmentProxy.h"
// #include "Public/Simulation/Misc/BezierCurveComponent.h"

// Sets default values
ADeepDriveJunctionProxy::ADeepDriveJunctionProxy()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

	Root = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
	SetRootComponent(Root);

	// m_BezierCurve = CreateDefaultSubobject<UBezierCurveComponent>(TEXT("BezierCurve"));
}

// Called when the game starts or when spawned
void ADeepDriveJunctionProxy::BeginPlay()
{
	Super::BeginPlay();

	m_IsGameRunning = true;
}

bool ADeepDriveJunctionProxy::ShouldTickIfViewportsOnly() const
{
	return true;
}

// Called every frame
void ADeepDriveJunctionProxy::Tick(float DeltaTime)
{
#if WITH_EDITOR

	Super::Tick(DeltaTime);

	if (!m_IsGameRunning)
	{
		uint8 prio = 100;

		TArray<FVector> points;
		for(auto &l : LinksIn)
			if(l)
				points.Add(l->getEndPoint());

		for(auto &l : LinksOut)
			if(l)
				points.Add(l->getStartPoint());

		if(points.Num() > 0)
		{
			FBox junctionBox(points);
			DrawDebugSolidBox(GetWorld(), junctionBox, JunctionColor, FTransform(), false, 0.0f, prio);
		}

		prio = 120;
		for(auto &lc : LaneConnections)
		{
			FVector fromStart;
			FVector fromEnd;
			FVector toStart;
			FVector toEnd;
			if (extractConnection(lc, fromStart, fromEnd, toStart, toEnd))
			{
				switch (lc.ConnectionShape)
				{
					case EDeepDriveConnectionShape::STRAIGHT_LINE:
						DrawDebugLine(GetWorld(), fromEnd, toStart, ConnectionColor, false, 0.0f, m_DrawPrioConnection, 10.0f);
						break;

					case EDeepDriveConnectionShape::QUADRATIC_SPLINE:
						drawQuadraticConnectionSegment(fromStart, fromEnd, toStart, toEnd);
						// carryOverDistance = addQuadraticConnectionSegment(m_RoadNetwork->Segments[segment.SegmentId], m_RoadNetwork->Segments[nextLink.Lanes[curLane].Segments[0]], connectionSegmentId, carryOverDistance);
						break;

					case EDeepDriveConnectionShape::CUBIC_SPLINE:
						drawCubicConnectionSegment(fromStart, fromEnd, toStart, toEnd, lc.CustomCurveParam);
						break;

					case EDeepDriveConnectionShape::ROAD_SEGMENT:
						// carryOverDistance = addSegmentToPoints(connectionSegment, false, carryOverDistance);
						break;

				}
			}
		}
	}

#endif
}

bool ADeepDriveJunctionProxy::extractConnection(const FDeepDriveLaneConnectionProxy &connectionProxy, FVector &fromStart, FVector &fromEnd, FVector &toStart, FVector &toEnd)
{
	const float delta = 500.0f;

	int32 found = 0;

	if(connectionProxy.FromSegment)
	{
		if	(	connectionProxy.FromSegment->getSpline()
			&&	connectionProxy.FromSegment->getSpline()->GetNumberOfSplinePoints() > 2
			)
		{
			const USplineComponent *spline = connectionProxy.FromSegment->getSpline();
			const float length = spline->GetSplineLength();

			fromStart = spline->GetLocationAtDistanceAlongSpline(FMath::Max(0.0f, length - delta), ESplineCoordinateSpace::World);
			fromEnd = spline->GetLocationAtDistanceAlongSpline(length, ESplineCoordinateSpace::World);
		}
		else
		{
			fromStart = connectionProxy.FromSegment->getStartPoint();
			fromEnd = connectionProxy.FromSegment->getEndPoint();
		}
		found |= 1;
	}
	else if(connectionProxy.FromLink)
	{
		fromStart = connectionProxy.FromLink->getStartPoint();
		fromEnd = connectionProxy.FromLink->getEndPoint();
		found |= 1;
	}

	if(connectionProxy.ToSegment)
	{
		if	(	connectionProxy.ToSegment->getSpline()
			&&	connectionProxy.ToSegment->getSpline()->GetNumberOfSplinePoints() > 2
			)
		{
			const USplineComponent *spline = connectionProxy.ToSegment->getSpline();
			const float length = spline->GetSplineLength();

			toStart = spline->GetLocationAtDistanceAlongSpline(0.0f, ESplineCoordinateSpace::World);
			toEnd = spline->GetLocationAtDistanceAlongSpline(FMath::Min(delta, length), ESplineCoordinateSpace::World);
		}
		else
		{
			toStart = connectionProxy.ToSegment->getStartPoint();
			toEnd = connectionProxy.ToSegment->getEndPoint();
		}
		found |= 2;
	}
	else if(connectionProxy.ToLink)
	{
		toStart = connectionProxy.ToLink->getStartPoint();
		toEnd = connectionProxy.ToLink->getEndPoint();
		found |= 2;
	}

	return found == 3;
}

void ADeepDriveJunctionProxy::drawQuadraticConnectionSegment(const FVector &fromStart, const FVector &fromEnd, const FVector &toStart, const FVector &toEnd)
{
	const FVector &p0 = fromEnd;
	FVector p1 = calcIntersectionPoint(fromStart, fromEnd, toStart, toEnd);
	const FVector &p2 = toStart;

	FVector a = p0;
	const float dT = 0.05f;
	for(float t = dT; t < 1.0f; t+= dT)
	{
		const float oneMinusT = (1.0f - t);
		const FVector b = oneMinusT * oneMinusT * p0 + 2.0f * oneMinusT * t * p1 + t * t *p2;

		DrawDebugLine(GetWorld(), a, b, ConnectionColor, false, 0.0f, m_DrawPrioConnection, 10.0f);

		a = b;
	}
	DrawDebugLine(GetWorld(), a, p2, ConnectionColor, false, 0.0f, m_DrawPrioConnection, 10.0f);

}

void ADeepDriveJunctionProxy::drawCubicConnectionSegment(const FVector &fromStart, const FVector &fromEnd, const FVector &toStart, const FVector &toEnd, const FVector &params)
{
	FVector dir0 = fromEnd - fromStart;
	FVector dir1 = toStart - toEnd;
	dir0.Normalize();
	dir1.Normalize();

	const FVector &p0 = fromEnd;
	const FVector &p3 = toStart;
	FVector p1 = p0 + dir0 * params.X;
	FVector p2 = p3 + dir1 * params.Y;

	FVector a = p0;
	const float dT = 0.05f;
	for(float t = dT; t < 1.0f; t+= dT)
	{
		const float oneMinusT = (1.0f - t);
		const FVector b = oneMinusT * oneMinusT * oneMinusT * p0 + (3.0f * oneMinusT * oneMinusT * t) * p1 + (3.0f * oneMinusT * t * t) * p2 + t * t * t * p3;

		DrawDebugLine(GetWorld(), a, b, ConnectionColor, false, 0.0f, m_DrawPrioConnection, 10.0f);

		a = b;
	}
	DrawDebugLine(GetWorld(), a, p3, ConnectionColor, false, 0.0f, m_DrawPrioConnection, 10.0f);
}


FVector ADeepDriveJunctionProxy::calcIntersectionPoint(const FVector &fromStart, const FVector &fromEnd, const FVector &toStart, const FVector &toEnd)
{
	FVector2D r = FVector2D(fromEnd - fromStart);
	FVector2D s = FVector2D(toStart - toEnd);

	float dot = FMath::Abs(FVector2D::DotProduct(r, s) / (r.Size() * s.Size()));

	//r.Normalize();
	//s.Normalize();

	float cRS = FVector2D::CrossProduct(r, s);

	if (FMath::Abs(cRS) > 0.001f && dot < 0.975f)
	{
		FVector2D qp(toEnd - fromStart);
		//qp.Normalize();
		float t = FVector2D::CrossProduct(qp, s) / cRS;
		FVector2D intersection(FVector2D(fromStart) + r * t);
		const float z = 0.5f * (toEnd.Z + fromStart.Z);

		return FVector(intersection, z);
	}
	return 0.5f * (fromEnd + toStart); 
}
