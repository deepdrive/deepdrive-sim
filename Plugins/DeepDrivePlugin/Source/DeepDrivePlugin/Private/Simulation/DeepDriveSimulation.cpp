

#include "DeepDrivePluginPrivatePCH.h"
#include "DeepDriveSimulation.h"

#include "Private/Server/DeepDriveServer.h"
#include "Public/Simulation/DeepDriveSimulationServerProxy.h"
#include "Public/Simulation/DeepDriveSimulationCaptureProxy.h"

#include "Public/Simulation/Agent/DeepDriveAgent.h"
#include "Public/Simulation/Agent/DeepDriveAgentControllerCreator.h"
#include "Public/Simulation/Misc/DeepDriveSimulationFreeCamera.h"

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
	
	m_curAgent = spawnAgent(InitialControllerMode);
	if(m_curAgent)
	{
		OnCurrentAgentChanged(m_curAgent);

		m_curAgentController = Cast<ADeepDriveAgentControllerBase> (m_curAgent->GetController());

		SelectCamera(EDeepDriveAgentCameraType::CHASE_CAMERA);

		const TSet < UActorComponent * > &components = GetComponents();
		for(auto &comp : components)
		{
			UCaptureSinkComponentBase *captureSinkComp = Cast<UCaptureSinkComponentBase> (comp);
			if(captureSinkComp)
			{
				m_CaptureSinks.Add(captureSinkComp);
				UE_LOG(LogDeepDriveSimulation, Log, TEXT("Found sink %s"), *(captureSinkComp->getName()));
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
		if (FreeCamera)
			FreeCamera->MoveForward(AxisValue);
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
		if (FreeCamera)
			FreeCamera->MoveRight(AxisValue);
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
			if(FreeCamera)
				FreeCamera->LookUp(AxisValue);
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
			if(FreeCamera)
				FreeCamera->Turn(AxisValue);
			break;

		case EDeepDriveAgentCameraType::ORBIT_CAMERA:
			m_OrbitCameraYaw += AxisValue;
			m_curAgent->SetOrbitCameraRotation(m_OrbitCameraPitch, m_OrbitCameraYaw);
			break;

	}
}

void ADeepDriveSimulation::SelectCamera(EDeepDriveAgentCameraType CameraType)
{
	if (CameraType != m_curCameraType)
	{
		APlayerCameraManager *cameraMgr = UGameplayStatics::GetPlayerCameraManager(GetWorld(), 0);
		if (cameraMgr)
		{
			AActor *camActor = 0;

			if (m_curAgent)
			{
				m_curAgent->ActivateCamera(CameraType);

				if (CameraType == EDeepDriveAgentCameraType::FREE_CAMERA)
				{
					if (FreeCamera)
					{
						FTransform agentTransform = m_curAgent->GetActorTransform();
						FreeCamera->SetActorTransform(FTransform(agentTransform.Rotator(), agentTransform.GetLocation() + FVector(0.0f, 0.0f, 200.0f), agentTransform.GetScale3D()));
						camActor = FreeCamera;
					}
				}
				else
				{
					camActor = m_curAgent;
				}

				if (camActor)
				{
					FViewTargetTransitionParams transitionParams;
					transitionParams.BlendTime = 0.0f;
					transitionParams.BlendFunction = VTBlend_Linear;
					transitionParams.BlendExp = 0.0f;
					transitionParams.bLockOutgoing = false;
					cameraMgr->SetViewTarget(camActor, transitionParams);

					m_curCameraType = CameraType;
				}
			}
		}
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

bool ADeepDriveSimulation::resetAgent()
{
	return m_curAgentController ? m_curAgentController->ResetAgent() : false;
}

ADeepDriveAgent* ADeepDriveSimulation::spawnAgent(EDeepDriveAgentControlMode mode)
{
	FTransform transform = GetActorTransform();
	ADeepDriveAgent *agent  = Cast<ADeepDriveAgent>(GetWorld()->SpawnActor(Agent, &transform, FActorSpawnParameters()));

	if(agent)
	{
		agent->setResetTransform(transform);

		ADeepDriveAgentControllerBase *controller = spawnController(mode);
		if(controller)
		{
			if(controller->Activate(*agent))
			{
				m_curAgentMode = mode;
				OnAgentSpawned(agent);
				UE_LOG(LogDeepDriveSimulation, Log, TEXT("Spawning agent controlled by %s"), *(controller->getControllerName()) );
			}
			else
				UE_LOG(LogDeepDriveSimulation, Log, TEXT("Couldn't activate controller %s"), *(controller->getControllerName()) );
		}
		else
			UE_LOG(LogDeepDriveSimulation, Log, TEXT("Couldn't spawn controller") );
	}
	else
		UE_LOG(LogDeepDriveSimulation, Log, TEXT("Couldn't spawn agent") );

	return agent;
}


ADeepDriveAgentControllerBase* ADeepDriveSimulation::spawnController(EDeepDriveAgentControlMode mode)
{
	ADeepDriveAgentControllerBase *controller = 0;

	controller = ControllerCreators.Contains(mode) ? ControllerCreators[mode]->CreateController() : 0;

	UE_LOG(LogDeepDriveSimulation, Log, TEXT("spawnController has it %c -> %p"), ControllerCreators.Contains(mode) ? 'T' :'F', controller );

	return controller;
}

void ADeepDriveSimulation::OnDebugTrigger()
{
	if (m_curAgent)
	{
		m_curAgent->OnDebugTrigger();
	}
}
