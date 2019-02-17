
#include "DeepDrivePluginPrivatePCH.h"
#include "DeepDriveJunctionProxy.h"

#include "Public/Simulation/RoadNetwork/DeepDriveRoadLinkProxy.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoadSegmentProxy.h"
#include "Public/Simulation/Misc/BezierCurveComponent.h"

// Sets default values
ADeepDriveJunctionProxy::ADeepDriveJunctionProxy()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

	Root = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
	SetRootComponent(Root);

	m_BezierCurve = CreateDefaultSubobject<UBezierCurveComponent>(TEXT("BezierCurve"));
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
					DrawDebugLine(GetWorld(), fromEnd, toStart, ConnectionColor, false, 0.0f, prio, 10.0f);
					break;
#if 0
				case EDeepDriveConnectionShape::QUADRATIC_SPLINE:
					carryOverDistance = addQuadraticConnectionSegment(m_RoadNetwork->Segments[segment.SegmentId], m_RoadNetwork->Segments[nextLink.Lanes[curLane].Segments[0]], connectionSegmentId, carryOverDistance);
					break;
				case EDeepDriveConnectionShape::CUBIC_SPLINE:
					carryOverDistance = addCubicConnectionSegment(m_RoadNetwork->Segments[segment.SegmentId], m_RoadNetwork->Segments[nextLink.Lanes[curLane].Segments[0]], connectionSegmentId, carryOverDistance);
					break;
				case EDeepDriveConnectionShape::ROAD_SEGMENT:
					carryOverDistance = addSegmentToPoints(connectionSegment, false, carryOverDistance);
					break;
#endif
				}
			}
		}
	}

#endif
}

bool ADeepDriveJunctionProxy::extractConnection(const FDeepDriveLaneConnectionProxy &connectionProxy, FVector &fromStart, FVector &fromEnd, FVector &toStart, FVector &toEnd)
{
	int32 found = 0;

	if(connectionProxy.FromSegment)
	{
		fromStart = connectionProxy.FromSegment->getStartPoint();
		fromEnd = connectionProxy.FromSegment->getEndPoint();
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
		toStart = connectionProxy.ToSegment->getStartPoint();
		toEnd = connectionProxy.ToSegment->getEndPoint();
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

