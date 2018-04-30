

#include "DeepDrivePluginPrivatePCH.h"
#include "DeepDriveSimulationFreeCamera.h"


// Sets default values
ADeepDriveSimulationFreeCamera::ADeepDriveSimulationFreeCamera()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

	FreeCamera = CreateDefaultSubobject<UCameraComponent>(TEXT("FreeCamera"));
	FreeCamera->SetupAttachment(GetRootComponent());

}

// Called when the game starts or when spawned
void ADeepDriveSimulationFreeCamera::BeginPlay()
{
	Super::BeginPlay();
	
}

// Called every frame
void ADeepDriveSimulationFreeCamera::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

}

void ADeepDriveSimulationFreeCamera::MoveForward(float AxisValue)
{
	SetActorLocation(GetActorLocation() + GetActorForwardVector() * AxisValue * ForwardSpeed);
}

void ADeepDriveSimulationFreeCamera::MoveRight(float AxisValue)
{
	SetActorLocation(GetActorLocation() + GetActorRightVector() * AxisValue * RightSpeed);
}

void ADeepDriveSimulationFreeCamera::LookUp(float AxisValue)
{
}

void ADeepDriveSimulationFreeCamera::Turn(float AxisValue)
{
}

