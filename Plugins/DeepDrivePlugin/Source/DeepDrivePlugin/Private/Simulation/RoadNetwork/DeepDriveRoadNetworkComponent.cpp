

#include "DeepDrivePluginPrivatePCH.h"
#include "DeepDriveRoadNetworkComponent.h"

#include "Public/Simulation/RoadNetwork/DeepDriveRoadLinkProxy.h"
#include "Public/Simulation/RoadNetwork/DeepDriveJunctionProxy.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoadSegmentProxy.h"


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

void UDeepDriveRoadNetworkComponent::PlaceAgentOnRoad(ADeepDriveAgent *Agent, bool RandomLocation)
{

}


void UDeepDriveRoadNetworkComponent::collectRoadNetwork()
{
	TMap<FString, int32> segmentMap;

	//	extract all segments (proxies) from scene
	TArray<AActor*> segments;
	UGameplayStatics::GetAllActorsOfClass(GetWorld(), TSubclassOf<ADeepDriveRoadSegmentProxy>(), segments);
	for(auto &actor : segments)
	{
		ADeepDriveRoadSegmentProxy* segmentProxy = Cast<ADeepDriveRoadSegmentProxy>(actor);
		if(segmentProxy)
		{
			SDeepDriveRoadSegment segment;

			segment.StartPoint = segmentProxy->getStartPoint();
			segment.EndPoint = segmentProxy->getEndPoint();

			const FSplineCurves *splineCurves = segmentProxy->getSplineCurves();
			if(splineCurves)
			{
				segment.Spline = new FSplineCurves;
				segment.Spline->Position = splineCurves->Position;
				segment.Spline->Rotation = splineCurves->Rotation;
				segment.Spline->Scale = splineCurves->Scale;
				segment.Spline->ReparamTable = splineCurves->ReparamTable;
			}

			m_RoadNetwork.Segments.Add(segment);
			segmentMap.Add(UKismetSystemLibrary::GetObjectName(segmentProxy), m_RoadNetwork.Segments.Num());
		}
	}

	TMap<FString, int32> linkMap;

	//	extract all road links (proxies) from scene
	TArray<AActor*> links;
	UGameplayStatics::GetAllActorsOfClass(GetWorld(), TSubclassOf<ADeepDriveRoadLinkProxy>(), links);
	for(auto &actor : links)
	{
		ADeepDriveRoadLinkProxy* linkProxy = Cast<ADeepDriveRoadLinkProxy>(actor);
		if(linkProxy)
		{
			SDeepDriveRoadLink link;
			link.StartPoint = linkProxy->getStartPoint();
			link.EndPoint = linkProxy->getEndPoint();

			for(auto &laneProxy : linkProxy->getLanes())
			{
				SDeepDriveLane lane;
				lane.LaneType = laneProxy.LaneType;

				for(auto &segProxy : laneProxy.Segments)
				{
					if(segmentMap.Contains(UKismetSystemLibrary::GetObjectName(segProxy)))
					{
						lane.Segments.Add( &m_RoadNetwork.Segments[segmentMap[UKismetSystemLibrary::GetObjectName(segProxy)]] );
					}

				}
				link.Lanes.Add(lane);
			}

			m_RoadNetwork.Links.Add(link);

			linkMap.Add(UKismetSystemLibrary::GetObjectName(linkProxy), m_RoadNetwork.Links.Num());
		}
	}

	// extract all junctions (proxies) from scene
	TArray<AActor*> junctions;
	UGameplayStatics::GetAllActorsOfClass(GetWorld(), TSubclassOf<ADeepDriveJunctionProxy>(), junctions);
	for(auto &actor : junctions)
	{
		ADeepDriveJunctionProxy* junctionProxy = Cast<ADeepDriveJunctionProxy>(actor);
		if(junctionProxy)
		{
			SDeepDriveJunction junction;

			for(auto &linkProxy : junctionProxy->getLinks())
			{
				if(linkMap.Contains(UKismetSystemLibrary::GetObjectName(linkProxy)))
				{
					junction.RoadLinks.Add( &m_RoadNetwork.Links[linkMap[UKismetSystemLibrary::GetObjectName(linkProxy)]] );
				}
			}

			for(auto &connectionProxy : junctionProxy->getLaneConnections())
			{
				SDeepDriveLaneConnection connection;

				if	(	segmentMap.Contains(UKismetSystemLibrary::GetObjectName(connectionProxy.FromSegment))
					&&	segmentMap.Contains(UKismetSystemLibrary::GetObjectName(connectionProxy.ToSegment))
					&&	(	connectionProxy.ConnectionSegment == 0
						|| 	segmentMap.Contains(UKismetSystemLibrary::GetObjectName(connectionProxy.ConnectionSegment))
						)
					)
				{
					connection.FromSegment = &m_RoadNetwork.Segments[segmentMap[UKismetSystemLibrary::GetObjectName(connectionProxy.FromSegment)]];
					connection.ToSegment = &m_RoadNetwork.Segments[segmentMap[UKismetSystemLibrary::GetObjectName(connectionProxy.ToSegment)]];
					if(connectionProxy.ConnectionSegment)
						connection.ConnectionSegment = &m_RoadNetwork.Segments[segmentMap[UKismetSystemLibrary::GetObjectName(connectionProxy.ConnectionSegment)]];
				}

				junction.Connections.Add(connection);
			}

			m_RoadNetwork.Junctions.Add(junction);
		}
	}

}
