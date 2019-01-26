

#include "DeepDrivePluginPrivatePCH.h"
#include "DeepDriveRoadNetworkComponent.h"

#include "Public/Simulation/RoadNetwork/DeepDriveRoadLinkProxy.h"
#include "Public/Simulation/RoadNetwork/DeepDriveJunctionProxy.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoadSegmentProxy.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoadNetworkExtractor.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoute.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"

#include "Private/Simulation/RoadNetwork/DeepDriveRouteCalculator.h"

DEFINE_LOG_CATEGORY(LogDeepDriveRoadNetwork);

// Sets default values for this component's properties
UDeepDriveRoadNetworkComponent::UDeepDriveRoadNetworkComponent()
{
	// Set this component to be initialized when the game starts, and to be ticked every frame.  You can turn these features
	// off to improve performance if you don't need them.
	PrimaryComponentTick.bCanEverTick = true;

	// ...
}


// Called when the game starts
void UDeepDriveRoadNetworkComponent::BeginPlay()
{
	Super::BeginPlay();

}

void UDeepDriveRoadNetworkComponent::Initialize()
{
	collectRoadNetwork();
}


// Called every frame
void UDeepDriveRoadNetworkComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	// ...
}

FVector UDeepDriveRoadNetworkComponent::GetRandomLocation(EDeepDriveLaneType PreferredLaneType, int32 relPos)
{
	FVector location = FVector::ZeroVector;

	const int32 numLinks = m_RoadNetwork.Links.Num();
	if(numLinks > 0)
	{
		SDeepDriveRoadLink &link = m_RoadNetwork.Links[FMath::RandRange(0u, numLinks - 1)];

		int32 laneInd = link.getRightMostLane(PreferredLaneType);
		if(laneInd)
		{
			SDeepDriveLane &lane = link.Lanes[laneInd];
			if(relPos == 0)
				location = m_RoadNetwork.Segments[lane.Segments[0]].StartPoint;
			else if(relPos > 0)
				location = m_RoadNetwork.Segments[lane.Segments[lane.Segments.Num() - 1]].EndPoint;
			else
				location = 0.5f * (m_RoadNetwork.Segments[lane.Segments[0]].StartPoint + m_RoadNetwork.Segments[lane.Segments[0]].EndPoint);
		}
	}

	return location;
}

void UDeepDriveRoadNetworkComponent::PlaceAgentOnRoad(ADeepDriveAgent *Agent, FVector Location)
{
	SDeepDriveRoadLink *link = UDeepDriveRoadNetworkComponent::findClosestLink(Location);
	if(link)
	{
		FTransform transform(FRotator(0.0f, link->Heading, 0.0f), link->StartPoint, FVector(1.0f, 1.0f, 1.0f));
		Agent->SetActorTransform(transform, false, 0, ETeleportType::TeleportPhysics);
	}
}

void UDeepDriveRoadNetworkComponent::PlaceAgentOnRoadRandomly(ADeepDriveAgent *Agent)
{
}

ADeepDriveRoute* UDeepDriveRoadNetworkComponent::CalculateRoute(const FVector Start, const FVector Destination)
{
	ADeepDriveRoute *route = 0;
	SDeepDriveRoadLink *startLink = findClosestLink(Start);
	SDeepDriveRoadLink *destLink = findClosestLink(Destination);

	if(startLink && destLink)
	{
		UE_LOG(LogDeepDriveRoadNetwork, Log, TEXT("Calc route from %d(%s) to %d(%s)"), startLink->LinkId, *(startLink->StartPoint.ToString()), destLink->LinkId, *(destLink->EndPoint.ToString()) );

		route = GetWorld()->SpawnActor<ADeepDriveRoute>(FVector(0.0f, 0.0f, 0.0f), FRotator(0.0f, 0.0f, 0.0f), FActorSpawnParameters());

		if(route)
		{
			DeepDriveRouteCalculator routeCalculator(m_RoadNetwork);

			SDeepDriveRouteData routeData = routeCalculator.calculate(Start, Destination);

			route->initialize(m_RoadNetwork, routeData);
		}
	}
	else
		UE_LOG(LogDeepDriveRoadNetwork, Log, TEXT("Calc route failed %p / %p"), startLink, destLink );

	return route;
}

ADeepDriveRoute* UDeepDriveRoadNetworkComponent::CalculateRoute(const TArray<uint32> &routeLinks)
{
	ADeepDriveRoute *route = 0;
	route = GetWorld()->SpawnActor<ADeepDriveRoute>(FVector(0.0f, 0.0f, 0.0f), FRotator(0.0f, 0.0f, 0.0f), FActorSpawnParameters());

	if (route)
	{
		SDeepDriveRouteData routeData;
		routeData.Start = FVector::ZeroVector;
		routeData.Destination = FVector::ZeroVector;
		routeData.Links = routeLinks;

		route->initialize(m_RoadNetwork, routeData);
	}
	return route;
}

SDeepDriveRoadLink* UDeepDriveRoadNetworkComponent::findClosestLink(const FVector &pos)
{
	uint32 key = 0;
	float dist = TNumericLimits<float>::Max();

	for (auto curIt = m_RoadNetwork.Links.CreateIterator(); curIt; ++curIt)
	{
		float curDist = (curIt.Value().StartPoint - pos).Size();
		if(curDist < dist)
		{
			dist = curDist;
			key = curIt.Key();
		}
		curDist = (curIt.Value().EndPoint - pos).Size();
		if(curDist < dist)
		{
			dist = curDist;
			key = curIt.Key();
		}
	}

	return key ? &m_RoadNetwork.Links[key] : 0;
}

void UDeepDriveRoadNetworkComponent::collectRoadNetwork()
{
	if(m_Extractor == 0)
		m_Extractor = new DeepDriveRoadNetworkExtractor(GetWorld());

	m_Extractor->extract(m_RoadNetwork);

	for(auto &rl : Route)
	{
		uint32 linkId = m_Extractor->getRoadLink(rl);
		if(linkId)
			m_DebugRoute.Add(linkId);
	}

	UE_LOG(LogDeepDriveRoadNetwork, Log, TEXT("Collected road network %d juntions %d links %d segments dgbRoute %d"), m_RoadNetwork.Junctions.Num(), m_RoadNetwork.Links.Num(), m_RoadNetwork.Segments.Num(), m_DebugRoute.Num() );
}

uint32 UDeepDriveRoadNetworkComponent::getRoadLink(ADeepDriveRoadLinkProxy *linkProxy)
{
	return m_Extractor ? m_Extractor->getRoadLink(linkProxy) : 0;
}

float UDeepDriveRoadNetworkComponent::calcHeading(const FVector &from, const FVector &to)
{
	FVector2D dir = FVector2D(to - from);
	dir.Normalize();
	return FMath::RadiansToDegrees(FMath::Atan2(dir.Y, dir.X));

}
