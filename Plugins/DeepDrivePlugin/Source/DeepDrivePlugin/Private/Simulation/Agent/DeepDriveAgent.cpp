
#include "DeepDrivePluginPrivatePCH.h"
#include "Public/Simulation/DeepDriveSimulation.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"
#include "Public/Capture/CaptureCameraComponent.h"
#include "Public/DeepDriveData.h"
#include "Components/SplineComponent.h"
#include "Simulation/Agent/DeepDriveAgentControllerBase.h"

#include "WheeledVehicleMovementComponent.h"
#include "Runtime/Engine/Classes/Kismet/KismetRenderingLibrary.h"

DEFINE_LOG_CATEGORY(LogDeepDriveAgent);

int32 ADeepDriveAgent::s_nextAgentId = 1;

ADeepDriveAgent::ADeepDriveAgent()
{
	m_AgentId = s_nextAgentId++;

	ChaseCameraStick = CreateDefaultSubobject<USpringArmComponent>(TEXT("ChaseCameraStick"));
	ChaseCameraStick->SetupAttachment(GetRootComponent());

	ChaseCamera = CreateDefaultSubobject<UCameraComponent>(TEXT("ChaseCamera"));
	ChaseCamera->SetupAttachment(ChaseCameraStick);

	InteriorCamera = CreateDefaultSubobject<UCameraComponent>(TEXT("InteriorCamera"));
	InteriorCamera->SetupAttachment(GetRootComponent());

	OrbitCameraArm = CreateDefaultSubobject<USpringArmComponent>(TEXT("OrbitCameraArm"));
	OrbitCameraArm->SetupAttachment(GetRootComponent());

	OrbitCamera = CreateDefaultSubobject<UCameraComponent>(TEXT("OrbitCamera"));
	OrbitCamera->SetupAttachment(OrbitCameraArm);

	CollisionRoot = CreateDefaultSubobject<USceneComponent>(TEXT("CollisionRoot"));
	CollisionRoot->SetupAttachment(GetRootComponent());

	UBoxComponent** boxes[] =	{ &CollisionFrontCenterBumper, &CollisionFrontLeftBumper, &CollisionFrontRightBumper, &CollisionFrontLeftFender, &CollisionFrontRightFender
								, &CollisionLeftDoor, &CollisionRightDoor, &CollisionRearCenterBumper
								, &CollisionRearLeftBumper, &CollisionRearRightBumper, &CollisionRearLeftFender, &CollisionRearRightFender
								};

	FName	names[] =	{	TEXT("CollisionFrontCnterBumper"), TEXT("CollisionFrontLeftBumper"), TEXT("CollisionFrontRightBumper"), TEXT("CollisionFrontLeftFender"), TEXT("CollisionFrontRightFender")
						,	TEXT("CollisionLeftDoor"), TEXT("CollisionRightDoor"), TEXT("CollisionRearCenterBumper")
						,	TEXT("CollisionRearLeftBumper"), TEXT("CollisionRearRightBumper"), TEXT("CollisionRearLeftFender"), TEXT("CollisionRearRightFender")
						};

	FName	tags[] =	{	TEXT("front_center_bumper"), TEXT("front_left_bumper"), TEXT("front_right_bumper"), TEXT("front_left_fender"), TEXT("front_right_fender")
						,	TEXT("left_door"), TEXT("right_door"), TEXT("rear_center_bumper")
						,	TEXT("rear_left_bumper"), TEXT("rear_right_bumper"), TEXT("rear_left_fender"), TEXT("rear_right_fender")
						};

	for (unsigned i = 0; i < sizeof(boxes) / sizeof(UBoxComponent**); ++i)
	{
		UBoxComponent *box = CreateDefaultSubobject<UBoxComponent>(names[i]);
		box->SetupAttachment(CollisionRoot);
		*(boxes[i]) = box;

		box->OnComponentBeginOverlap.AddDynamic(this, &ADeepDriveAgent::OnBeginOverlap);
		box->ComponentTags.Add(tags[i]);
	}
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

	if(m_AgentController == 0)
		m_AgentController = Cast<ADeepDriveAgentControllerBase>(GetController());

	if(m_AgentController && m_AgentController->updateAgentOnTrack())
	{
		++m_NumberOfLaps;
		UE_LOG(LogDeepDriveAgent, Log, TEXT("Laps finished %d"), m_NumberOfLaps );
	}
}

int32 ADeepDriveAgent::RegisterCaptureCamera(float fieldOfView, int32 captureWidth, int32 captureHeight, FVector relativePosition, FVector relativeRotation, const FString &label)
{
	int32 camId = 0;

	UTextureRenderTarget2D *sceneTexture = UKismetRenderingLibrary::CreateRenderTarget2D(GetWorld(), captureWidth, captureHeight, ETextureRenderTargetFormat::RTF_RGBA16f);
	UTextureRenderTarget2D *depthTexture = UKismetRenderingLibrary::CreateRenderTarget2D(GetWorld(), captureWidth, captureHeight, ETextureRenderTargetFormat::RTF_R16f);

	if(sceneTexture && depthTexture)
	{
		UCaptureCameraComponent *captureCamCmp = NewObject<UCaptureCameraComponent>(this);
		if(captureCamCmp)
		{
			captureCamCmp->SetupAttachment(RootComponent);
			captureCamCmp->SetRelativeLocation(relativePosition);
			captureCamCmp->SetRelativeRotation(FRotator(relativeRotation.Y, relativeRotation.Z, relativeRotation.X));
			captureCamCmp->RegisterComponent();

			captureCamCmp->Initialize(sceneTexture, depthTexture, fieldOfView);

			camId = captureCamCmp->getCameraId();
			const int32 camIndex = m_CaptureCameras.Num();
			m_CaptureCameras.Add(captureCamCmp);
			OnCaptureCameraAdded(camId, camIndex, sceneTexture, label);

		}
	}

	return camId;
}

bool ADeepDriveAgent::setViewMode(int32 cameraId, const FString &viewModeName)
{
	bool res = false;

	const FDeepDriveViewMode *viewMode = 0;

	
	if (viewModeName.IsEmpty() == false)
	{
		if (m_Simulation && m_Simulation->ViewModes.Contains(viewModeName))
		{
			viewMode = &m_Simulation->ViewModes[viewModeName];
			res = true;
		}
		else
			UE_LOG(LogDeepDriveAgent, Error, TEXT("ViewMode %s not found"), *(viewModeName) );
	}
	else
		res = true;

	if (res)
	{
		if (cameraId < 0)
		{
			for (int32 camIndex = 0; camIndex < m_CaptureCameras.Num(); ++camIndex)
			{
				m_CaptureCameras[camIndex]->setViewMode(viewMode);
				SetDepthTexture(cameraId, camIndex, viewMode ? m_CaptureCameras[camIndex]->getDepthRenderTexture() : 0);
			}
		}
		else
		{
			int32 camIndex = findCaptureCamera(cameraId);
			UE_LOG(LogDeepDriveAgent, Log, TEXT("Camera index %d"), camIndex );
			if (camIndex >= 0)
			{
				m_CaptureCameras[camIndex]->setViewMode(viewMode);
				SetDepthTexture(cameraId, camIndex, viewMode ? m_CaptureCameras[camIndex]->getDepthRenderTexture() : 0);
			}
			else
			{
				res = false;
				UE_LOG(LogDeepDriveAgent, Error, TEXT("Camera with id %d not found"), cameraId );
			}
		}
	}

	return res;
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
}

void ADeepDriveAgent::OnLapFinished()
{
}

void ADeepDriveAgent::reset()
{
	m_NumberOfLaps = 0;
	GetVehicleMovementComponent()->SetThrottleInput(0.0f);
	GetVehicleMovementComponent()->SetSteeringInput(0.0f);
	GetVehicleMovementComponent()->SetBrakeInput(1.0f);
	GetMesh()->SetAllPhysicsLinearVelocity(FVector::ZeroVector);
}

void ADeepDriveAgent::beginCapture(DeepDriveDataOut &deepDriveData)
{
	deepDriveData.Position = GetActorLocation();
	deepDriveData.Rotation = GetActorRotation();
	deepDriveData.Velocity = GetVelocity();
	deepDriveData.AngularVelocity = m_AngularVelocity;
	deepDriveData.Acceleration = m_Acceleration;
	deepDriveData.AngularAcceleration = m_AngularAcceleration;
	deepDriveData.Speed = getSpeed();

	deepDriveData.Dimension = m_Dimensions;

	deepDriveData.IsGameDriving = m_isGameDriving;

	deepDriveData.Steering = m_curSteering;
	deepDriveData.Throttle = m_curThrottle;
	deepDriveData.Brake = m_curBrake;
	deepDriveData.Handbrake = m_curHandbrake;


	deepDriveData.LapNumber = m_NumberOfLaps;

	ADeepDriveAgentControllerBase *ctrl = getAgentController();
	if (ctrl)
	{
		ctrl->getCollisionData(deepDriveData.CollisionData);
	    deepDriveData.DistanceAlongRoute = ctrl->getDistanceAlongRoute();
	    deepDriveData.RouteLength = ctrl->getRouteLength();
	    deepDriveData.DistanceToCenterOfLane = ctrl->getDistanceToCenterOfTrack();
	}
	else
	{
	    deepDriveData.CollisionData = DeepDriveCollisionData();
	}
}

float ADeepDriveAgent::getSpeed() const
{
	return GetVehicleMovementComponent()->GetForwardSpeed();
}

float ADeepDriveAgent::getSpeedKmh() const
{
	return GetVehicleMovementComponent()->GetForwardSpeed() * 0.036f;
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


ADeepDriveAgentControllerBase *ADeepDriveAgent::getAgentController()
{
	return Cast<ADeepDriveAgentControllerBase>(GetController());
}

int32 ADeepDriveAgent::findCaptureCamera(int32 id)
{
	int32 index = 0;
	for (index = 0; index < m_CaptureCameras.Num(); ++index)
	{
		if (m_CaptureCameras[index]->getCameraId() == id)
			break;
	}
	return index < m_CaptureCameras.Num() ? index : -1;
}

void ADeepDriveAgent::OnBeginOverlap(UPrimitiveComponent *OverlappedComponent, AActor *OtherActor, UPrimitiveComponent *OtherComp, int32 OtherBodyIndex, bool bFromSweep, const FHitResult &SweepResult)
{
	if(OtherActor != this)
	{
		UE_LOG(LogDeepDriveAgent, Log, TEXT("OnBeginOverlap %s with %s"), *(OverlappedComponent->GetName()), *(OtherActor->GetName()) );

		ADeepDriveAgentControllerBase *ctrl = getAgentController();
		if (ctrl)
		{
			ctrl->OnAgentCollision(OtherActor, SweepResult, OverlappedComponent->ComponentTags.Num() > 0 ? OverlappedComponent->ComponentTags[0] : FName());
		}
	}
}
