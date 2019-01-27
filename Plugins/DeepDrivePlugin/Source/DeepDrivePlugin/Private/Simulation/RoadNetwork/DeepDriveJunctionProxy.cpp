
#include "DeepDrivePluginPrivatePCH.h"
#include "DeepDriveJunctionProxy.h"

#include "Public/Simulation/RoadNetwork/DeepDriveRoadLinkProxy.h"

// Sets default values
ADeepDriveJunctionProxy::ADeepDriveJunctionProxy()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

	Root = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
	SetRootComponent(Root);


}

// Called when the game starts or when spawned
void ADeepDriveJunctionProxy::BeginPlay()
{
	Super::BeginPlay();

	m_IsGameRunning = true;
}

// Called every frame
void ADeepDriveJunctionProxy::Tick(float DeltaTime)
{
#if WITH_EDITOR

	Super::Tick(DeltaTime);

	if (!m_IsGameRunning)
	{
		const uint8 prio = 220;

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
			DrawDebugSolidBox(GetWorld(), junctionBox, Color, FTransform(), false, 0.0f, prio);
		}
	}

#endif
}


bool ADeepDriveJunctionProxy::ShouldTickIfViewportsOnly() const
{
	return true;
}

