

#include "DeepDrivePluginPrivatePCH.h"
#include "DeepDriveSimulation.h"

#include "Private/Server/DeepDriveServer.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"

#include "Public/Simulation/Agent/Controllers/DeepDriveAgentManualController.h"

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
	
	m_curAgent = spawnAgent(EDeepDriveAgentControlMode::DDACM_MANUAL);

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
	if (m_curAgentController)
	{
		m_curAgentController->MoveForward(AxisValue);
	}
}

void ADeepDriveSimulation::MoveRight(float AxisValue)
{
	if (m_curAgentController)
	{
		m_curAgentController->MoveRight(AxisValue);
	}
}

void ADeepDriveSimulation::LookUp(float AxisValue)
{
	
}

void ADeepDriveSimulation::Turn(float AxisValue)
{
	
}

void ADeepDriveSimulation::OnCameraSelect(EDeepDriveAgentCameraType CameraType)
{
	
}

void ADeepDriveSimulation::OnSelectMode(EDeepDriveAgentControlMode Mode)
{
	
}

ADeepDriveAgent* ADeepDriveSimulation::spawnAgent(EDeepDriveAgentControlMode mode)
{
	FTransform transform(FVector(-28210.0f, 22480.0f, 22716.0f));
	ADeepDriveAgent *agent  = Cast<ADeepDriveAgent>(GetWorld()->SpawnActor(Agent, &transform, FActorSpawnParameters()));

	ADeepDriveAgentControllerBase *controller = spawnController(mode);

	if (controller)
	{
		m_curAgentController = controller;
		controller->Possess(agent);
	}
	UE_LOG(LogDeepDriveSimulation, Log, TEXT("Spawning agent %p"), agent );

	return agent;
}


ADeepDriveAgentControllerBase* ADeepDriveSimulation::spawnController(EDeepDriveAgentControlMode mode)
{
	FTransform transform;
	ADeepDriveAgentControllerBase *controller = Cast<ADeepDriveAgentControllerBase>(GetWorld()->SpawnActor(ADeepDriveAgentManualController::StaticClass(), &transform, FActorSpawnParameters()));
	return controller;
}
