#include "UtilsProxy.h"

#include "Runtime/Landscape/Classes/LandscapeSplineSegment.h"

//DEFINE_LOG_CATEGORY(LogDeepDriveUtilsProxy);

// Sets default values
AUtilsProxy::AUtilsProxy()
{
 	// Set this actor to call Tick() every frame. You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = false;
}

int32 AUtilsProxy::GetSplineSegmentConnections(UObject* StructProperty)
{
	UE_LOG(LogTemp, Log, TEXT("GetSplineSegmentConnections seg: %d"), StructProperty);
	return 42;
}

// Called when the game starts or when spawned
void AUtilsProxy::BeginPlay()
{
	Super::BeginPlay();
	UE_LOG(LogTemp, Log, TEXT("UtilsProxy started"));
}

// Called every frame
void AUtilsProxy::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

}

