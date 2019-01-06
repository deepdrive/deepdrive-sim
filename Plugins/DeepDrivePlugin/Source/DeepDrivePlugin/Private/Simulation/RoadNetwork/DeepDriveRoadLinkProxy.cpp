
#include "DeepDrivePluginPrivatePCH.h"
#include "DeepDriveRoadLinkProxy.h"


// Sets default values
ADeepDriveRoadLinkProxy::ADeepDriveRoadLinkProxy()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

	Root = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
	SetRootComponent(Root);

	StartPoint = CreateDefaultSubobject<UArrowComponent>(TEXT("StartPoint"));
	StartPoint->SetupAttachment(Root);

	EndPoint = CreateDefaultSubobject<UArrowComponent>(TEXT("EndPoint"));
	EndPoint->SetupAttachment(Root);

}

// Called when the game starts or when spawned
void ADeepDriveRoadLinkProxy::BeginPlay()
{
	Super::BeginPlay();

	m_IsGameRunning = true;
	
}

// Called every frame
void ADeepDriveRoadLinkProxy::Tick(float DeltaTime)
{
#if WITH_EDITOR

	Super::Tick(DeltaTime);

	if (!m_IsGameRunning)
	{
		const uint8 prio = 200;
		DrawDebugLine(GetWorld(), StartPoint->GetComponentLocation(), EndPoint->GetComponentLocation(), Color, false, 0.0f, prio, 8.0f);
	}

#endif

}

bool ADeepDriveRoadLinkProxy::ShouldTickIfViewportsOnly() const
{
	return true;
}

