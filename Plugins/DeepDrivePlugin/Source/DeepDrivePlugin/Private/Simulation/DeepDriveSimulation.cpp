

#include "DeepDrivePluginPrivatePCH.h"
#include "DeepDriveSimulation.h"

#include "Private/Server/DeepDriveServer.h"
#include "Public/Simulation/DeepDriveSimulationServerProxy.h"
#include "Public/Simulation/DeepDriveSimulationCaptureProxy.h"

#include "Public/Simulation/Agent/DeepDriveAgent.h"
#include "Public/Simulation/Agent/DeepDriveAgentControllerCreator.h"

#include "Public/CaptureSink/CaptureSinkComponentBase.h"

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
		m_ServerProxy = new DeepDriveSimulationServerProxy(*this);
		m_CaptureProxy = new DeepDriveSimulationCaptureProxy(*this, CaptureInterval);
		if	(	m_ServerProxy && m_ServerProxy->initialize(IPAddress, Port)
			&&	m_CaptureProxy
			)
		{
			m_isActive = true;
			UE_LOG(LogDeepDriveSimulation, Log, TEXT("DeepDriveSimulation [%s] activated"), *(GetFullName()));
		}
		else
		{
			UE_LOG(LogDeepDriveSimulation, Error, TEXT("DeepDriveSimulation [%s] could not be activated ServerProxy %p"), *(GetFullName()), m_ServerProxy);
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

		const TSet < UActorComponent * > &components = GetComponents();
		for(auto &comp : components)
		{
			UCaptureSinkComponentBase *captureSinkComp = Cast<UCaptureSinkComponentBase> (comp);
			if(captureSinkComp)
			{
				m_CaptureSinks.Add(captureSinkComp);
				UE_LOG(LogDeepDriveCapture, Log, TEXT("Found sink %s"), *(captureSinkComp->getName()));
			}
		}

	}

}

void ADeepDriveSimulation::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	Super::EndPlay(EndPlayReason);

	if (m_isActive)
	{
		if(m_CaptureProxy)
			m_CaptureProxy->shutdown();

		if(m_ServerProxy)
			m_ServerProxy->shutdown();

		m_isActive = false;
		UE_LOG(LogDeepDriveSimulation, Log, TEXT("DeepDriveSimulation [%s] unregistered"), *(GetFullName()));
	}
}


// Called every frame
void ADeepDriveSimulation::Tick( float DeltaTime )
{
	Super::Tick( DeltaTime );

	if (m_isActive)
	{
		if(m_CaptureProxy)
			m_CaptureProxy->update(DeltaTime);

		if(m_ServerProxy)
			m_ServerProxy->update(DeltaTime);
	}
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
		ADeepDriveAgentControllerBase *controller = spawnController(Mode);
		ADeepDriveAgentControllerBase *prevController = Cast<ADeepDriveAgentControllerBase> (m_curAgent->GetController());

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
	ADeepDriveAgentControllerBase *controller = 0;

	controller = ControllerCreators.Contains(mode) ? ControllerCreators[mode]->CreateController() : 0;

	UE_LOG(LogDeepDriveSimulation, Log, TEXT("spawnController has it %c -> %p"), ControllerCreators.Contains(mode) ? 'T' :'F', controller );

	return controller;
}
