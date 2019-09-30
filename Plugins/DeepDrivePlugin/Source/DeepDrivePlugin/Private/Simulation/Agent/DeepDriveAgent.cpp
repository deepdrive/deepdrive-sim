
#include "DeepDrivePluginPrivatePCH.h"
#include "Public/Simulation/DeepDriveSimulation.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"
#include "Public/Capture/CaptureCameraComponent.h"
#include "Public/DeepDriveData.h"
#include "Components/SplineComponent.h"
#include "Simulation/Agent/DeepDriveAgentControllerBase.h"
#include "Private/Capture/DeepDriveCapture.h"

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

	CollisionSimpleBox = CreateDefaultSubobject<UBoxComponent>(TEXT("CollisionSimpleBox"));
	CollisionSimpleBox->SetupAttachment(CollisionRoot);
	// CollisionSimpleBox->OnComponentBeginOverlap.AddDynamic(this, &ADeepDriveAgent::OnBeginOverlap);
	CollisionSimpleBox->ComponentTags.Add(TEXT("simple"));

	UBoxComponent** boxes[] =	{ &CollisionFrontCenterBumper, &CollisionFrontLeftBumper, &CollisionFrontRightBumper, &CollisionFrontLeftFender, &CollisionFrontRightFender
								, &CollisionLeftDoor, &CollisionRightDoor, &CollisionRearCenterBumper
								, &CollisionRearLeftBumper, &CollisionRearRightBumper, &CollisionRearLeftFender, &CollisionRearRightFender
								};

	FName	names[] =	{	TEXT("CollisionFrontCenterBumper"), TEXT("CollisionFrontLeftBumper"), TEXT("CollisionFrontRightBumper"), TEXT("CollisionFrontLeftFender"), TEXT("CollisionFrontRightFender")
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

	if (	FrontBumperDistance <= 0.0f && BackBumperDistance <= 0.0f
		&&	CollisionFrontCenterBumper && CollisionRearCenterBumper
		)
	{
		FVector2D forwardBumperLoc(CollisionFrontCenterBumper->GetRelativeTransform().GetLocation());
		FVector2D rearBumperLoc(CollisionRearCenterBumper->GetRelativeTransform().GetLocation());
		FVector2D forward = forwardBumperLoc - rearBumperLoc;
		forward.Normalize();

		FrontBumperDistance = FMath::Abs(FVector2D::DotProduct(forward, forwardBumperLoc));
		BackBumperDistance = FMath::Abs(FVector2D::DotProduct(forward, rearBumperLoc));

		UE_LOG(LogDeepDriveAgent, Log, TEXT("Calculated FrontBumperDist to %f and BackBumperDistance to %f"), FrontBumperDistance, BackBumperDistance);
	}
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
			m_CaptureCameras.Add(camId, captureCamCmp);
			OnCaptureCameraAdded(camId, sceneTexture, label);
			SetCaptureEncoding(camId, EDeepDriveInternalCaptureEncoding::RGB_DEPTH, 0);

			if(m_Simulation && m_isEgoAgent == false)
				m_Simulation->onEgoAgentChanged(true);
			m_isEgoAgent = true;
		}
	}

	return camId;
}

void ADeepDriveAgent::UnregisterCaptureCamera(uint32 camId)
{
	if(camId)
	{
		if(m_CaptureCameras.Contains(camId))
		{
			UCaptureCameraComponent *camComp = m_CaptureCameras[camId];
			if(camComp)
			{
				OnCaptureCameraRemoved(camComp->getCameraId());
				DeepDriveCapture::GetInstance().UnregisterCaptureComponent(camComp->getCameraId());

				camComp->destroy();
				m_CaptureCameras[camId] = 0;
			}
			m_CaptureCameras.Remove(camId);
		}
		else
			UE_LOG(LogDeepDriveAgent, Error, TEXT("Camera with id % not found on agent %s"), camId, *(GetName()) );
	}
	else
	{
		UE_LOG(LogDeepDriveAgent, Log, TEXT("Unregistering all cameras on agent %s"), *(GetName()) );
		for(auto &cIt : m_CaptureCameras)
		{
			UCaptureCameraComponent *camComp = cIt.Value;
			if(camComp)
			{
				OnCaptureCameraRemoved(camComp->getCameraId());
				DeepDriveCapture::GetInstance().UnregisterCaptureComponent(camComp->getCameraId());

				camComp->destroy();
				cIt.Value = 0;
			}
		}
		m_CaptureCameras.Empty();
	}
	
	m_isEgoAgent = m_CaptureCameras.Num() > 0;
	if(m_Simulation && m_isEgoAgent == false)
		m_Simulation->onEgoAgentChanged(false);
}

void ADeepDriveAgent::setCollisionMode(bool simple)
{
	m_SimpleCollisionMode = simple;
	setCollisionVisibility(m_CollisionVisible);
}

void ADeepDriveAgent::setCollisionVisibility(bool visible)
{
	UBoxComponent* boxes[] =	{ CollisionFrontCenterBumper, CollisionFrontLeftBumper, CollisionFrontRightBumper, CollisionFrontLeftFender, CollisionFrontRightFender
								, CollisionLeftDoor, CollisionRightDoor, CollisionRearCenterBumper
								, CollisionRearLeftBumper, CollisionRearRightBumper, CollisionRearLeftFender, CollisionRearRightFender
								};

	if(visible)
	{
		if(m_SimpleCollisionMode)
		{
			for (unsigned i = 0; i < sizeof(boxes) / sizeof(UBoxComponent*); ++i)
				if(boxes[i])
					boxes[i]->SetHiddenInGame(true, false);
			CollisionSimpleBox->SetHiddenInGame(false, false);
		}
		else
		{
			for (unsigned i = 0; i < sizeof(boxes) / sizeof(UBoxComponent*); ++i)
				if(boxes[i])
					boxes[i]->SetHiddenInGame(false, false);
			CollisionSimpleBox->SetHiddenInGame(true, false);
		}
	}
	else
	{
		for (unsigned i = 0; i < sizeof(boxes) / sizeof(UBoxComponent*); ++i)
			if(boxes[i])
				boxes[i]->SetHiddenInGame(true, false);
		CollisionSimpleBox->SetHiddenInGame(true, false);
	}

	m_CollisionVisible = visible;
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
			for (auto &cIt :  m_CaptureCameras)
			{
				cIt.Value->setViewMode(viewMode);
				// SetDepthTexture(cIt.Value->getCameraId(), viewMode ? cIt.Value->getDepthRenderTexture() : 0);
				SetCaptureEncoding(cIt.Value->getCameraId(), viewMode ? viewMode->ViewModeEncoding : EDeepDriveInternalCaptureEncoding::RGB_DEPTH, viewMode ? cIt.Value->getDepthRenderTexture() : 0);
			}
		}
		else
		{
			if (m_CaptureCameras.Contains(cameraId))
			{
				m_CaptureCameras[cameraId]->setViewMode(viewMode);
				// SetDepthTexture(cameraId, viewMode ? m_CaptureCameras[cameraId]->getDepthRenderTexture() : 0);
				SetCaptureEncoding(cameraId, viewMode ? viewMode->ViewModeEncoding : EDeepDriveInternalCaptureEncoding::RGB_DEPTH, viewMode ? m_CaptureCameras[cameraId]->getDepthRenderTexture() : 0);
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
	OnReverseLight(GetVehicleMovementComponent()->GetForwardSpeed() < 0.0f && throttle < 0.0f);
}

void ADeepDriveAgent::SetBrake(float brake)
{
	m_curBrake = brake;
	OnBrakeLight(brake > 0.0f);
	GetVehicleMovementComponent()->SetBrakeInput(brake);
}

void ADeepDriveAgent::SetHandbrake(bool on)
{
	m_curHandbrake = on;
}

void ADeepDriveAgent::ActivateCamera(EDeepDriveAgentCameraType cameraType)
{
	m_hasFocus = true;
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
	m_hasFocus = false;

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
	GetVehicleMovementComponent()->StopMovementImmediately();
	GetMesh()->SetAllPhysicsLinearVelocity(FVector::ZeroVector);
	GetMesh()->SetPhysicsAngularVelocityInDegrees(FVector::ZeroVector);

	OnSimulationReset();
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

	deepDriveData.DistanceToNextAgent = m_NextAgent ? m_DistanceToNextAgent : -1.0f;
	deepDriveData.DistanceToPrevAgent = m_PrevAgent ? m_DistanceToPrevAgent : -1.0f;
	deepDriveData.DistanceToNextOpposingAgent = -1.0f;

	deepDriveData.LapNumber = m_NumberOfLaps;

	ADeepDriveAgentControllerBase *ctrl = getAgentController();
	if (ctrl)
	{
		ctrl->getCollisionData(deepDriveData.CollisionData);
	    deepDriveData.DistanceAlongRoute = ctrl->getDistanceAlongRoute();
	    deepDriveData.RouteLength = ctrl->getRouteLength();
	    deepDriveData.DistanceToCenterOfLane = ctrl->getDistanceToCenterOfTrack();
		deepDriveData.IsPassing = ctrl->isPassing();
	}
	else
	{
	    deepDriveData.CollisionData = DeepDriveCollisionData();
		deepDriveData.IsPassing = false;
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

void ADeepDriveAgent::OnBeginOverlap(UPrimitiveComponent *OverlappedComponent, AActor *OtherActor, UPrimitiveComponent *OtherComp, int32 OtherBodyIndex, bool bFromSweep, const FHitResult &SweepResult)
{
	if(OtherActor != this)
	{

		if	(	(m_SimpleCollisionMode && OverlappedComponent == CollisionSimpleBox)
			||	(!m_SimpleCollisionMode && OverlappedComponent != CollisionSimpleBox)
			)
		{
			// UE_LOG(LogDeepDriveAgent, Log, TEXT("OnBeginOverlap %s with %s"), *(OverlappedComponent->GetName()), *(OtherActor->GetName()) );

			ADeepDriveAgentControllerBase *ctrl = getAgentController();
			if (ctrl)
			{
				ctrl->OnAgentCollision(OtherActor, SweepResult, OverlappedComponent->ComponentTags.Num() > 0 ? OverlappedComponent->ComponentTags[0] : FName());
			}
		}
	}
}

EDeepDriveAgentState ADeepDriveAgent::GetAgentState()
{
	return EDeepDriveAgentState::CRUSING;
}

bool ADeepDriveAgent::IsEgoAgent()
{
	return m_isEgoAgent || (m_hasFocus && m_Simulation->hasEgoAgent() == false);
}

void ADeepDriveAgent::SetSpeedRange(float MinSpeed, float MaxSpeed)
{
	ADeepDriveAgentControllerBase *ctrl = getAgentController();
	if(ctrl)
		ctrl->SetSpeedRange(MinSpeed, MaxSpeed);
}

void ADeepDriveAgent::SetDirectionIndicatorState(EDeepDriveAgentDirectionIndicatorState DirectionIndicator)
{
	m_DirectionIndicator = DirectionIndicator;
	UE_LOG(LogDeepDriveAgent, Log, TEXT("SetDirectionIndicatorState %d"), static_cast<int32> (m_DirectionIndicator) );
}

EDeepDriveAgentDirectionIndicatorState ADeepDriveAgent::GetDirectionIndicatorState()
{
	return m_DirectionIndicator;
}
