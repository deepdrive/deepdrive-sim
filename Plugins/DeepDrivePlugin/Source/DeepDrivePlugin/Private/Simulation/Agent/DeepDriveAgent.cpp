
#include "DeepDrivePluginPrivatePCH.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"
#include "Public/Capture/CaptureCameraComponent.h"
#include "Components/SplineComponent.h"
#include "Simulation/Agent/DeepDriveAgentControllerBase.h"

#include "WheeledVehicleMovementComponent.h"
#include "Runtime/Engine/Classes/Kismet/KismetRenderingLibrary.h"

DEFINE_LOG_CATEGORY(LogDeepDriveAgent);

int32 ADeepDriveAgent::s_nextAgentId = 1;

ADeepDriveAgent::ADeepDriveAgent()
{
	m_AgentId = s_nextAgentId++;
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
	Super::Tick(DeltaSeconds);

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

			OnCaptureCameraAdded(targetTexture, label);

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
	GetVehicleMovementComponent()->SetBrakeInput(brake);
}

void ADeepDriveAgent::SetHandbrake(bool on)
{
	m_curHandbrake = on;
}

void ADeepDriveAgent::ActivateCamera(EDeepDriveAgentCameraType cameraType)
{
	ChaseCamera->Deactivate();
	InteriorCamera->Deactivate();
	OrbitCamera->Deactivate();

	switch(cameraType)
	{
		case EDeepDriveAgentCameraType::FREE_CAMERA:
			SetInstrumentsVisibility(false);
			break;

		case EDeepDriveAgentCameraType::CHASE_CAMERA:
			ChaseCamera->Activate();
			SetInstrumentsVisibility(true);
			break;

		case EDeepDriveAgentCameraType::INTERIOR_CAMERA:
			InteriorCamera->Activate();
			SetInstrumentsVisibility(false);
			break;

		case EDeepDriveAgentCameraType::ORBIT_CAMERA:
			OrbitCamera->Activate();
			SetInstrumentsVisibility(false);
			break;
	}
}

void ADeepDriveAgent::DeactivateCameras()
{
	ChaseCamera->Deactivate();
	InteriorCamera->Deactivate();
	OrbitCamera->Deactivate();
}

void ADeepDriveAgent::SetCenterOfTrackSpline(USplineComponent *Spline)
{
	m_CenterOfTrackSpline = Spline;
}

void ADeepDriveAgent::SetOrbitCameraRotation(float pitch, float yaw)
{
	OrbitCameraArm->SetRelativeRotation(FRotator(pitch, yaw, 0.0f));
}

void ADeepDriveAgent::OnLapStart()
{
	m_LapStarted = true;
}

void ADeepDriveAgent::OnLapFinished()
{
	if(m_LapStarted)
	{
		++m_NumberOfLaps;
		UE_LOG(LogDeepDriveAgent, Log, TEXT("Laps finished %d"), m_NumberOfLaps );
		m_LapStarted = false;
	}
}

void ADeepDriveAgent::reset()
{
	UE_LOG(LogDeepDriveAgent, Log, TEXT("Try to reset agent to %s"), *(m_ResetTransform.ToString()));
	SetActorTransform(m_ResetTransform, false, 0, ETeleportType::TeleportPhysics);

	SetControlValues(0.0f, 0.0f, 0.0f, false);
}

float ADeepDriveAgent::getSpeed() const
{
	return GetVehicleMovementComponent()->GetForwardSpeed();
}

float ADeepDriveAgent::getDistanceAlongRoute() const
{
	float res = 0.0f;
	if (m_CenterOfTrackSpline)
	{
		const float closestKey = m_CenterOfTrackSpline->FindInputKeyClosestToWorldLocation(GetActorLocation());

		const int32 index0 = floor(closestKey);
		const int32 index1 = floor(closestKey + 1.0f);

		const float dist0 = m_CenterOfTrackSpline->GetDistanceAlongSplineAtSplinePoint(index0);
		const float dist1 = m_CenterOfTrackSpline->GetDistanceAlongSplineAtSplinePoint(index1);

		res = FMath::Lerp(dist0, dist1, closestKey - static_cast<float> (index0));
	}
	return res;
}

float ADeepDriveAgent::getDistanceToCenterOfTrack() const
{
	float res = 0.0f;
	if (m_CenterOfTrackSpline)
	{
		FVector curLoc = GetActorLocation();
		const float closestKey = m_CenterOfTrackSpline->FindInputKeyClosestToWorldLocation(curLoc);
		res = (m_CenterOfTrackSpline->GetLocationAtSplineInputKey(closestKey, ESplineCoordinateSpace::World) - curLoc).Size();
	}
	return res;
}

float ADeepDriveAgent::getDistanceToObstacleAhead(float maxDistance)
{
	float distance = -1.0f;

	return distance;
}


void ADeepDriveAgent::OnCheckpointReached()
{
	ADeepDriveAgentControllerBase *ctrl = Cast<ADeepDriveAgentControllerBase>(GetController());
	if (ctrl)
		ctrl->OnCheckpointReached();
}

void ADeepDriveAgent::OnDebugTrigger()
{
	ADeepDriveAgentControllerBase *ctrl = Cast<ADeepDriveAgentControllerBase>(GetController());
	if (ctrl)
		ctrl->OnDebugTrigger();
}
