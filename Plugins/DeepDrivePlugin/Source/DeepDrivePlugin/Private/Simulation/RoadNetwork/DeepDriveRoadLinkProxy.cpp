
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
	
}

// Called every frame
void ADeepDriveRoadLinkProxy::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

}

