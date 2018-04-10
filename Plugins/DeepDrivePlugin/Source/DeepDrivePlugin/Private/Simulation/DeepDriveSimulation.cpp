

#include "DeepDrivePluginPrivatePCH.h"
#include "DeepDriveSimulation.h"

#include "Private/Server/DeepDriveServer.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"

#include "Public/Simulation/Agent/Controllers/DeepDriveAgentManualController.h"
#include "Public/Simulation/Agent/Controllers/DeepDriveAgentSplineController.h"

DEFINE_LOG_CATEGORY(LogDeepDriveSimulation);

// Sets default values
ADeepDriveSimulation::ADeepDriveSimulation()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

}

void ADeepDriveSimulation::PreInitializeComponents()
{
	Super::PreInitializeComponents();

	bool alreadyRegistered = false;
	TArray<AActor*> actors;
	UGameplayStatics::GetAllActorsOfClass(GetWorld(), TSubclassOf<ADeepDriveSimulation>(), actors);
	for (auto &actor : actors)
	{
		ADeepDriveSimulation *simu = Cast<ADeepDriveSimulation>(actor);
		if (	simu
			&&	simu != this
			)
		{
			if (simu->m_isActive)
			{
				alreadyRegistered = true;
				UE_LOG(LogDeepDriveSimulation, Log, TEXT("Another Simulation [%s] is already registered"), *(simu->GetFullName()));
				break;
			}
		}
	}

	if (!alreadyRegistered)
	{
		if (DeepDriveServer::GetInstance().RegisterProxy(*this))
		{
			m_isActive = true;
			UE_LOG(LogDeepDriveServerProxy, Log, TEXT("Server Proxy [%s] registered"), *(GetFullName()));
		}
	}

}


// Called when the game starts or when spawned
void ADeepDriveSimulation::BeginPlay()
{
	Super::BeginPlay();
	
	m_curAgent = spawnAgent(EDeepDriveAgentControlMode::MANUAL);
	if(m_curAgent)
	{
		m_curAgentController = Cast<ADeepDriveAgentControllerBase> (m_curAgent->GetController());

		m_curCameraType = EDeepDriveAgentCameraType::CHASE_CAMERA;
		m_curAgent->ActivateCamera(m_curCameraType);

	}

}

void ADeepDriveSimulation::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	Super::EndPlay(EndPlayReason);

	if (m_isActive)
	{
		DeepDriveServer::GetInstance().UnregisterProxy(*this);
		m_isActive = false;
		UE_LOG(LogDeepDriveSimulation, Log, TEXT("Server Proxy [%s] unregistered"), *(GetFullName()));
	}
}


// Called every frame
void ADeepDriveSimulation::Tick( float DeltaTime )
{
	Super::Tick( DeltaTime );

	if (m_isActive)
	{
		DeepDriveServer::GetInstance().update(DeltaTime);
	}
}

void ADeepDriveSimulation::RegisterClient(int32 ClientId, bool IsMaster)
{
}

void ADeepDriveSimulation::UnregisterClient(int32 ClientId, bool IsMaster)
{
}

int32 ADeepDriveSimulation::RegisterCaptureCamera(float FieldOfView, int32 CaptureWidth, int32 CaptureHeight, FVector RelativePosition, FVector RelativeRotation, const FString &Label)
{
	int32 camId = 0;

	return camId;
}

bool ADeepDriveSimulation::RequestAgentControl()
{
	bool res = false;

	return res;
}

void ADeepDriveSimulation::ReleaseAgentControl()
{
}

void ADeepDriveSimulation::ResetAgent()
{
}

void ADeepDriveSimulation::SetAgentControlValues(float steering, float throttle, float brake, bool handbrake)
{
}


void ADeepDriveSimulation::MoveForward(float AxisValue)
{
	if(m_curCameraType == EDeepDriveAgentCameraType::FREE_CAMERA)
	{
	
	}
	else if (m_curAgentController)
	{
		m_curAgentController->MoveForward(AxisValue);
	}
}

void ADeepDriveSimulation::MoveRight(float AxisValue)
{
	if(m_curCameraType == EDeepDriveAgentCameraType::FREE_CAMERA)
	{
	
	}
	else if (m_curAgentController)
	{
		m_curAgentController->MoveRight(AxisValue);
	}
}

void ADeepDriveSimulation::LookUp(float AxisValue)
{
	switch(m_curCameraType)
	{
		case EDeepDriveAgentCameraType::FREE_CAMERA:
			//FreeCamera->LookUp();
			break;

		case EDeepDriveAgentCameraType::ORBIT_CAMERA:
			m_OrbitCameraPitch += AxisValue;
			m_curAgent->SetOrbitCameraRotation(m_OrbitCameraPitch, m_OrbitCameraYaw);
			break;

	}
}

void ADeepDriveSimulation::Turn(float AxisValue)
{
	switch(m_curCameraType)
	{
		case EDeepDriveAgentCameraType::FREE_CAMERA:
			//FreeCamera->LookUp();
			break;

		case EDeepDriveAgentCameraType::ORBIT_CAMERA:
			m_OrbitCameraYaw += AxisValue;
			m_curAgent->SetOrbitCameraRotation(m_OrbitCameraPitch, m_OrbitCameraYaw);
			break;

	}
}

void ADeepDriveSimulation::SelectCamera(EDeepDriveAgentCameraType CameraType)
{
	if(CameraType != m_curCameraType)
	{
		m_curCameraType = CameraType;

		if(m_curCameraType != EDeepDriveAgentCameraType::FREE_CAMERA)
			m_curAgent->ActivateCamera(m_curCameraType);
	}
}

void ADeepDriveSimulation::SelectMode(EDeepDriveAgentControlMode Mode)
{
	if(Mode != m_curAgentMode)
	{
		UE_LOG(LogDeepDriveSimulation, Log, TEXT(">>>>>>>>>>>>>>>>>>>>< Switching Controllers %d %d <<<<<<<<<<<<<<<<<<<<"), m_curAgentMode, Mode  );

		ADeepDriveAgentControllerBase *controller = spawnController(Mode);
		ADeepDriveAgentControllerBase *prevController = Cast<ADeepDriveAgentControllerBase> (m_curAgent->GetController());

		UE_LOG(LogDeepDriveSimulation, Log, TEXT("PrevController %p <-> %p"), m_curAgentController, prevController  );

		if	(	controller
			&&	controller->Activate(*m_curAgent)
			)
		{
			if(prevController)
			{
				prevController->Deactivate();
				prevController->Destroy();
			}

			m_curAgentMode = Mode;
			m_curAgentController = controller;
			UE_LOG(LogDeepDriveSimulation, Log, TEXT("Switching to Controller %p <-> %p"), controller, m_curAgent->GetController() );
		}

	}
}

ADeepDriveAgent* ADeepDriveSimulation::spawnAgent(EDeepDriveAgentControlMode mode)
{
	FTransform transform = GetActorTransform();
	ADeepDriveAgent *agent  = Cast<ADeepDriveAgent>(GetWorld()->SpawnActor(Agent, &transform, FActorSpawnParameters()));

	ADeepDriveAgentControllerBase *controller = spawnController(mode);

	if	(	controller
		&&	controller->Activate(*agent)
		)
	{
		m_curAgentMode = mode;
		UE_LOG(LogDeepDriveSimulation, Log, TEXT("Spawning agent %p Controller %p"), agent, controller );
	}

	return agent;
}


ADeepDriveAgentControllerBase* ADeepDriveSimulation::spawnController(EDeepDriveAgentControlMode mode)
{
	FTransform transform;

	ADeepDriveAgentControllerBase *controller = 0;

	switch(mode)
	{
		case EDeepDriveAgentControlMode::MANUAL:
			controller = Cast<ADeepDriveAgentControllerBase>(GetWorld()->SpawnActor(ADeepDriveAgentManualController::StaticClass(), &transform, FActorSpawnParameters()));
			break;

		case EDeepDriveAgentControlMode::SPLINE:
			controller = Cast<ADeepDriveAgentControllerBase>(GetWorld()->SpawnActor(ADeepDriveAgentSplineController::StaticClass(), &transform, FActorSpawnParameters()));
			break;
	}

	return controller;
}
