
#include "DeepDrivePluginPrivatePCH.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"
#include "Public/Capture/CaptureCameraComponent.h"

#include "WheeledVehicleMovementComponent.h"
#include "Runtime/Engine/Classes/Kismet/KismetRenderingLibrary.h"

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

void ADeepDriveAgent::BeginPlay()
{
	Super::BeginPlay();

	FVector origin;
	GetActorBounds(true, origin, m_Dimensions);

	m_prevVelocity = m_AngularVelocity = m_prevAngularVelocity = FVector(0.0f, 0.0f, 0.0f);
}

void ADeepDriveAgent::Tick( float DeltaSeconds )
{
	FVector curVelocity = GetVelocity();
	m_Acceleration = (curVelocity - m_prevVelocity) / DeltaSeconds;
	m_prevVelocity = curVelocity;

	if(GetMesh())
	{
		m_AngularVelocity = GetMesh()->GetPhysicsAngularVelocityInDegrees();
		m_AngularAcceleration = (m_AngularVelocity - m_prevVelocity) / DeltaSeconds;
		m_prevAngularVelocity = m_AngularAcceleration;
	}
}

int32 ADeepDriveAgent::RegisterCaptureCamera(float fieldOfView, int32 captureWidth, int32 captureHeight, FVector relativePosition, FVector relativeRotation, const FString &label)
{
	int32 camId = 0;

	UTextureRenderTarget2D* targetTexture = UKismetRenderingLibrary::CreateRenderTarget2D(GetWorld(), captureWidth, captureHeight, ETextureRenderTargetFormat::RTF_RGBA16f);

	if(targetTexture)
	{
		UCaptureCameraComponent *captureCamCmp = NewObject<UCaptureCameraComponent>(this);
		if(captureCamCmp)
		{
			captureCamCmp->SetupAttachment(RootComponent);
			captureCamCmp->SetRelativeLocation(relativePosition);
			captureCamCmp->SetRelativeRotation(FRotator(relativeRotation.Y, relativeRotation.Z, relativeRotation.X));
			captureCamCmp->RegisterComponent();

			captureCamCmp->Initialize(targetTexture, fieldOfView);

			m_CaptureCameras.Add(captureCamCmp);

			camId = captureCamCmp->getCameraId();
		}
	}

	return camId;
}

void ADeepDriveAgent::SetControlValues(float steering, float throttle, float brake, bool handbrake)
{
	SetSteering(steering);
	SetThrottle(throttle);
	SetBrake(brake);
	SetHandbrake(handbrake);
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

void ADeepDriveAgent::SetBrake(float brake)
{
	m_curBrake = brake;
}

void ADeepDriveAgent::SetHandbrake(bool on)
{
	m_curHandbrake = on;
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


float ADeepDriveAgent::getSpeed() const
{
	return GetVehicleMovementComponent()->GetForwardSpeed();
}
