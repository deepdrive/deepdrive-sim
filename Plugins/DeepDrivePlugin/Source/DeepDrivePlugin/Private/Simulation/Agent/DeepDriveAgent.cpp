
#include "DeepDrivePluginPrivatePCH.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"

#include "Vehicles/WheeledVehicleMovementComponent.h"

DEFINE_LOG_CATEGORY(LogDeepDriveAgent);

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
	APlayerCameraManager *cameraMgr = UGameplayStatics::GetPlayerCameraManager(GetWorld(), 0);
	if(cameraMgr)
	{
		ChaseCamera->Deactivate();
		InteriorCamera->Deactivate();
		OrbitCamera->Deactivate();

		switch(cameraType)
		{
			case EDeepDriveAgentCameraType::CHASE_CAMERA:
				ChaseCamera->Activate();
				break;
			case EDeepDriveAgentCameraType::INTERIOR_CAMERA:
				InteriorCamera->Activate();
				break;
			case EDeepDriveAgentCameraType::ORBIT_CAMERA:
				OrbitCamera->Activate();
				break;
		}

		FViewTargetTransitionParams transitionParams;
		transitionParams.BlendTime = 0.0f;
		transitionParams.BlendFunction = VTBlend_Linear;
		transitionParams.BlendExp = 0.0f;
		transitionParams.bLockOutgoing = false;

		cameraMgr->SetViewTarget(this, transitionParams);
	}
}


void ADeepDriveAgent::SetOrbitCameraRotation(float pitch, float yaw)
{
	OrbitCameraArm->SetRelativeRotation(FRotator(pitch, yaw, 0.0f));
}
