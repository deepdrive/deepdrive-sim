

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
	TMap<FString, int32> linkMap;

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
					SDeepDriveRoadSegment segment;

					segment.StartPoint = segProxy->getStartPoint();
					segment.EndPoint = segProxy->getEndPoint();

					const FSplineCurves *splineCurves = segProxy->getSplineCurves();
					if(splineCurves)
					{
						segment.Spline = new FSplineCurves;
						segment.Spline->Position = splineCurves->Position;
						segment.Spline->Rotation = splineCurves->Rotation;
						segment.Spline->Scale = splineCurves->Scale;
						segment.Spline->ReparamTable = splineCurves->ReparamTable;
					}

					lane.Segments.Add(segment);
				}
				link.Lanes.Add(lane);
			}

			m_RoadNetwork.Links.Add(link);

			linkMap.Add(UKismetSystemLibrary::GetObjectName(linkProxy), m_RoadNetwork.Links.Num());
		}
	}


	TArray<AActor*> junctions;
	UGameplayStatics::GetAllActorsOfClass(GetWorld(), TSubclassOf<ADeepDriveJunctionProxy>(), junctions);
	for(auto &actor : junctions)
	{
		ADeepDriveJunctionProxy* junctionProxy = Cast<ADeepDriveJunctionProxy>(actor);
		if(junctionProxy)
		{
		}
	}

}
