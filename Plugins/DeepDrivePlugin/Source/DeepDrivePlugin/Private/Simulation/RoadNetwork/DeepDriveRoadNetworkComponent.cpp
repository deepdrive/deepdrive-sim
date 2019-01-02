

#include "DeepDrivePluginPrivatePCH.h"
#include "DeepDriveRoadNetworkComponent.h"

#include "Public/Simulation/RoadNetwork/DeepDriveRoadLinkProxy.h"
#include "Public/Simulation/RoadNetwork/DeepDriveJunctionProxy.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoadSegmentProxy.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoadNetworkExtractor.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoute.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"

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

FVector UDeepDriveRoadNetworkComponent::GetRandomLocation(EDeepDriveLaneType PreferredLaneType)
{
	FVector location;

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
	if (m_DebugRoute.Num() > 0)
	{	
		route = GetWorld()->SpawnActor<ADeepDriveRoute>(FVector(0.0f, 0.0f, 0.0f), FRotator(0.0f, 0.0f, 0.0f), FActorSpawnParameters());;

		if(route)
		{
			SDeepDriveRouteData routeData;
			routeData.Start = m_RoadNetwork.Links[m_DebugRoute[0]].StartPoint;
			routeData.Destination = m_RoadNetwork.Links[m_DebugRoute[m_DebugRoute.Num() - 1]].EndPoint;
			routeData.Links = m_DebugRoute;

			route->initialize(m_RoadNetwork, routeData);
		}
	}
	return route;
}

SDeepDriveRoadLink* UDeepDriveRoadNetworkComponent::findClosestLink(const FVector &pos)
{
	uint32 key = 0;
	float dist = 1000000.0f;

	for (auto curIt = m_RoadNetwork.Links.CreateIterator(); curIt; ++curIt)
	{
		const float curDist = (curIt.Value().StartPoint - pos).Size();
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
	DeepDriveRoadNetworkExtractor extractor(GetWorld());

	extractor.extract(m_RoadNetwork);

	for(auto &rl : Route)
	{
		uint32 linkId = extractor.getRoadLink(rl);
		if(linkId)
			m_DebugRoute.Add(linkId);
	}

	UE_LOG(LogDeepDriveRoadNetwork, Log, TEXT("Collected road network %d juntions %d links %d segments dgbRoute %d"), m_RoadNetwork.Junctions.Num(), m_RoadNetwork.Links.Num(), m_RoadNetwork.Segments.Num(), m_DebugRoute.Num() );
}

float UDeepDriveRoadNetworkComponent::calcHeading(const FVector &from, const FVector &to)
{
	FVector2D dir = FVector2D(to - from);
	dir.Normalize();
	return FMath::RadiansToDegrees(FMath::Atan2(dir.Y, dir.X));

}
