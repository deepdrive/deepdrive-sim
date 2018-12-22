

#include "DeepDrivePluginPrivatePCH.h"
#include "DeepDrivePlugin.h"
#include "DeepDriveRoadSegmentProxy.h"
#include "Components/SplineComponent.h"


// Sets default values
ADeepDriveRoadSegmentProxy::ADeepDriveRoadSegmentProxy()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

	Root = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
	SetRootComponent(Root);

	StartPoint = CreateDefaultSubobject<UArrowComponent>(TEXT("StartPoint"));
	StartPoint->SetupAttachment(Root);

	EndPoint = CreateDefaultSubobject<UArrowComponent>(TEXT("EndPoint"));
	EndPoint->SetupAttachment(Root);

	Spline = CreateDefaultSubobject<USplineComponent>(TEXT("Spline"));
	Spline->SetupAttachment(Root);

}

// Called when the game starts or when spawned
void ADeepDriveRoadSegmentProxy::BeginPlay()
{
	Super::BeginPlay();
	
}

// Called every frame
void ADeepDriveRoadSegmentProxy::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

}

