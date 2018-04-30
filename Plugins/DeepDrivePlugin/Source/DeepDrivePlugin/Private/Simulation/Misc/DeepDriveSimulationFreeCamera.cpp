

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

	const FRotator curRotation = GetActorRotation();
	FRotator newRotation(FMath::ClampAngle(curRotation.Pitch - m_LookUp, -85.0, 85.0), curRotation.Yaw + m_Turn, 0.0f);

	m_curForward = FMath::FInterpTo(m_curForward, m_DesiredForward, DeltaTime, InterpolationSpeed);
	m_curRight = FMath::FInterpTo(m_curRight, m_DesiredRight, DeltaTime, InterpolationSpeed);

	const FVector forward = GetActorForwardVector();
	const FVector right = GetActorRightVector();

	FVector newLocation = GetActorLocation() + forward * m_curForward * ForwardSpeed + right * m_curRight * RightSpeed;

	SetActorTransform(FTransform(newRotation, newLocation, FVector(1.0f, 1.0f, 1.0f)));
}

