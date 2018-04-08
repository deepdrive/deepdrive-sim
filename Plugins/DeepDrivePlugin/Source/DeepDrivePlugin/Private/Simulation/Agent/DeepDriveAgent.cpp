
#include "DeepDrivePluginPrivatePCH.h"
#include "DeepDriveAgent.h"

#include "Vehicles/WheeledVehicleMovementComponent.h"


ADeepDriveAgent::ADeepDriveAgent()
{
	ChaseCamera = CreateDefaultSubobject<UCameraComponent>(TEXT("ChaseCamera"));
	ChaseCamera->SetupAttachment(GetRootComponent());

	InteriorCamera = CreateDefaultSubobject<UCameraComponent>(TEXT("InteriorCamera"));
	InteriorCamera->SetupAttachment(GetRootComponent());

	OrbitCameraArm = CreateDefaultSubobject<USpringArmComponent>(TEXT("OrbitCameraArm"));
	OrbitCameraArm->SetupAttachment(GetRootComponent());

	OrbitCamera = CreateDefaultSubobject<UCameraComponent>(TEXT("OrbitCamera"));
	OrbitCamera->SetupAttachment(OrbitCameraArm);

}


void ADeepDriveAgent::RegisterCaptureCamera(float fieldOfView, int32 captureWidth, int32 captureHeight, FVector relativePosition, FVector relativeRotation, const FString &label)
{
}

void ADeepDriveAgent::SetControlValues(float steering, float throttle, float brake, bool handbrake)
{
}

void ADeepDriveAgent::SetSteering(float steering)
{
	m_curSteering = steering;
	GetVehicleMovementComponent()->SetSteeringInput(steering);
}

void ADeepDriveAgent::SetThrottle(float throttle)
{
	m_curThrottle = throttle;
	GetVehicleMovementComponent()->SetThrottleInput(throttle);
}

void ADeepDriveAgent::ActivateCamera(EDeepDriveAgentCameraType cameraType)
{
}
