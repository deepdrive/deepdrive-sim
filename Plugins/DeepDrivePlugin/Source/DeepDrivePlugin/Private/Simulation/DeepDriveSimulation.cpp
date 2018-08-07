

#include "DeepDrivePluginPrivatePCH.h"
#include "DeepDriveSimulation.h"
#include "Private/Server/DeepDriveSimulationServer.h"
#include "Private/Simulation/DeepDriveSimulationMessageHandler.h"
#include "Private/Simulation/DeepDriveSimulationStateMachine.h"
#include "Private/Simulation/States/DeepDriveSimulationInitializeState.h"
#include "Private/Simulation/States/DeepDriveSimulationRunningState.h"
#include "Private/Simulation/States/DeepDriveSimulationResetState.h"

#include "Private/Server/DeepDriveServer.h"
#include "Public/Simulation/DeepDriveSimulationServerProxy.h"
#include "Public/Simulation/DeepDriveSimulationCaptureProxy.h"

#include "Public/Simulation/DeepDriveSimulationTypes.h"

#include "Public/Simulation/Agent/DeepDriveAgent.h"
#include "Public/Simulation/Agent/DeepDriveAgentControllerCreator.h"
#include "Public/Simulation/Misc/DeepDriveSimulationFreeCamera.h"
#include "Public/Simulation/Misc/DeepDriveRandomStream.h"

#include "Public/CaptureSink/CaptureSinkComponentBase.h"

DEFINE_LOG_CATEGORY(LogDeepDriveSimulation);

// Sets default values
ADeepDriveSimulation::ADeepDriveSimulation()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

}

ADeepDriveSimulation::~ADeepDriveSimulation()
{

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
			if (simu->isActive())
			{
				alreadyRegistered = true;
				UE_LOG(LogDeepDriveSimulation, Log, TEXT("Another Simulation [%s] is already registered"), *(simu->GetFullName()));
				break;
			}
		}
	}

	if (!alreadyRegistered)
	{
		int32 ipParts[4];
		if	(	DeepDriveServer::convertIpAddress(SimulationIPAddress, ipParts)
			&&	SimulationPort > 0 && SimulationPort <= 65535
			)
		{
			m_SimulationServer = new DeepDriveSimulationServer(*this, ipParts, static_cast<uint16> (SimulationPort));
			if (m_SimulationServer)
			{
				m_SimulationServer->start();
				m_MessageHandler = new DeepDriveSimulationMessageHandler(*this, *m_SimulationServer);
			}
		}

		m_ServerProxy = new DeepDriveSimulationServerProxy(*this);
		m_CaptureProxy = new DeepDriveSimulationCaptureProxy(*this, CaptureInterval);
		if	(	m_ServerProxy && m_ServerProxy->initialize(ClientsIPAddress, ClientsPort, GetWorld())
			&&	m_CaptureProxy
			)
		{
			m_StateMachine = new DeepDriveSimulationStateMachine();

			if (m_StateMachine)
			{
				m_StateMachine->registerState(new DeepDriveSimulationInitializeState(*m_StateMachine));
				m_StateMachine->registerState(new DeepDriveSimulationRunningState(*m_StateMachine));
				m_StateMachine->registerState(new DeepDriveSimulationResetState(*m_StateMachine));
			}

			for (auto &rsd : RandomStreams)
			{
				UDeepDriveRandomStream *randomStream = NewObject<UDeepDriveRandomStream>();
				if (randomStream)
				{
					randomStream->initialize(Seed);
					rsd.Value.setRandomStream(randomStream);
				}
			}

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

	if (m_StateMachine)
	{
		SetTickableWhenPaused(true);
		m_StateMachine->setNextState("Initialize");
	}
}

void ADeepDriveSimulation::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	Super::EndPlay(EndPlayReason);

	if (isActive())
	{
		if (m_SimulationServer)
			m_SimulationServer->Stop();

		if(m_CaptureProxy)
			m_CaptureProxy->shutdown();

		if(m_ServerProxy)
			m_ServerProxy->shutdown();

		UE_LOG(LogDeepDriveSimulation, Log, TEXT("DeepDriveSimulation [%s] unregistered"), *(GetFullName()));
	}
}


// Called every frame
void ADeepDriveSimulation::Tick( float DeltaTime )
{
	Super::Tick( DeltaTime );

	if(m_MessageHandler)
		m_MessageHandler->handleMessages();

	if (m_StateMachine)
		m_StateMachine->update(*this, DeltaTime);
}

void ADeepDriveSimulation::ResetSimulation()
{
	if(m_StateMachine)
		m_StateMachine->setNextState("Reset");
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

void ADeepDriveSimulation::switchToCamera(EDeepDriveAgentCameraType type)
{
	APlayerCameraManager *cameraMgr = UGameplayStatics::GetPlayerCameraManager(GetWorld(), 0);
	if (cameraMgr)
	{
		AActor *camActor = 0;

		if (m_curAgent)
		{
			m_curAgent->ActivateCamera(type);

			if (type == EDeepDriveAgentCameraType::FREE_CAMERA)
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

				m_curCameraType = type;
			}
		}
	}
}

void ADeepDriveSimulation::SelectMode(EDeepDriveAgentControlMode Mode)
{
	if(Mode != m_curAgentMode)
	{
		ADeepDriveAgentControllerBase *controller = spawnController(Mode, 0, 0);
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

void ADeepDriveSimulation::NextAgent()
{
	switchToAgent((m_curAgentIndex + 1) % m_Agents.Num());
}

void ADeepDriveSimulation::PreviousAgent()
{
	int32 index = m_curAgentIndex - 1;
	if(index < 0)
		index = m_Agents.Num() - 1;

	switchToAgent(index);
}

void ADeepDriveSimulation::enqueueMessage(deepdrive::server::MessageHeader *message)
{
	if (message)
	{
	 	if(m_MessageHandler)
	 		m_MessageHandler->enqueueMessage(*message);
	 	else
			FMemory::Free(message);
	}
}

void ADeepDriveSimulation::applyGraphicsSettings(const SimulationGraphicsSettings &gfxSettings)
{
	UGameUserSettings *gameSettings = UGameUserSettings::GetGameUserSettings();
	if (gameSettings)
	{
		gameSettings->SetFullscreenMode(gfxSettings.is_fullscreen > 0 ? EWindowMode::Fullscreen : EWindowMode::Windowed);
		gameSettings->SetVSyncEnabled(gfxSettings.vsync_enabled > 0);

		gameSettings->SetResolutionScaleNormalized(gfxSettings.resolution_scale);

		gameSettings->SetTextureQuality(gfxSettings.texture_quality);
		gameSettings->SetShadowQuality(gfxSettings.shadow_quality);
		gameSettings->SetVisualEffectQuality(gfxSettings.effect_quality);		
		gameSettings->SetPostProcessingQuality(gfxSettings.post_process_level);
		gameSettings->SetViewDistanceQuality(gfxSettings.view_distance);

		UE_LOG(LogDeepDriveSimulation, Log, TEXT("DeepDriveSimulation Texture Quality %d"), gfxSettings.texture_quality);

		gameSettings->ApplySettings(false);

	}
}

bool ADeepDriveSimulation::resetAgent()
{
	return m_curAgentController ? m_curAgentController->ResetAgent() : false;
}

void ADeepDriveSimulation::initializeAgents()
{
	m_curAgent = spawnAgent(InitialControllerMode, InitialConfigurationSlot, StartPositionSlot);
	if (m_curAgent)
	{
		OnCurrentAgentChanged(m_curAgent);

		m_curAgentController = Cast<ADeepDriveAgentControllerBase>(m_curAgent->GetController());

		SelectCamera(EDeepDriveAgentCameraType::CHASE_CAMERA);

		const TSet < UActorComponent * > &components = GetComponents();
		for (auto &comp : components)
		{
			UCaptureSinkComponentBase *captureSinkComp = Cast<UCaptureSinkComponentBase>(comp);
			if (captureSinkComp)
			{
				m_CaptureSinks.Add(captureSinkComp);
				UE_LOG(LogDeepDriveSimulation, Log, TEXT("Found sink %s"), *(captureSinkComp->getName()));
			}
		}
	}
}

ADeepDriveAgent* ADeepDriveSimulation::spawnAgent(EDeepDriveAgentControlMode mode, int32 configSlot, int32 startPosSlot)
{
	FTransform transform = GetActorTransform();
	ADeepDriveAgent *agent  = Cast<ADeepDriveAgent>(GetWorld()->SpawnActor(Agent, &transform, FActorSpawnParameters()));

	if(agent)
	{
		m_Agents.Add(agent);
		agent->initialize(*this);
		agent->setResetTransform(transform);

		ADeepDriveAgentControllerBase *controller = spawnController(mode, configSlot, startPosSlot);
		if(controller)
		{
			if(controller->Activate(*agent))
			{
				m_curAgentMode = mode;
				OnAgentSpawned(agent);
				UE_LOG(LogDeepDriveSimulation, Log, TEXT("Spawning agent controlled by %s"), *(controller->getControllerName()) );

				spawnAdditionalAgents();
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

void ADeepDriveSimulation::spawnAdditionalAgents()
{
	for (auto &data : AdditionalAgents)
	{
		FTransform transform;
		ADeepDriveAgent *agent = Cast<ADeepDriveAgent>(GetWorld()->SpawnActor(data.Agent, &transform, FActorSpawnParameters()));

		if (agent)
		{
			m_Agents.Add(agent);
			agent->setResetTransform(transform);

			ADeepDriveAgentControllerBase *controller = spawnController(data.Mode, data.ConfigurationSlot, data.StartPositionSlot);
			if (controller)
			{
				if (controller->Activate(*agent))
				{
					OnAgentSpawned(agent);
					UE_LOG(LogDeepDriveSimulation, Log, TEXT("Additional agent spawned, controlled by %s"), *(controller->getControllerName()));
				}
				else
					UE_LOG(LogDeepDriveSimulation, Log, TEXT("Couldn't activate controller %s"), *(controller->getControllerName()));
			}
			else
				UE_LOG(LogDeepDriveSimulation, Log, TEXT("Couldn't spawn controller for additional agent"));
		}
		else
			UE_LOG(LogDeepDriveSimulation, Log, TEXT("Couldn't spawn additional agent"));

	}
}

ADeepDriveAgentControllerBase* ADeepDriveSimulation::spawnController(EDeepDriveAgentControlMode mode, int32 configSlot, int32 startPosSlot)
{
	ADeepDriveAgentControllerBase *controller = 0;

	controller = ControllerCreators.Contains(mode) ? ControllerCreators[mode]->CreateAgentController(configSlot, startPosSlot, this) : 0;

	UE_LOG(LogDeepDriveSimulation, Log, TEXT("spawnController has it %c -> %p"), ControllerCreators.Contains(mode) ? 'T' :'F', controller );

	return controller;
}

void ADeepDriveSimulation::switchToAgent(int32 index)
{
	if(index >= 0 && index < m_Agents.Num())
	{
		m_curAgent->DeactivateCameras();
		m_curAgent = m_Agents[index];
		switchToCamera(m_curCameraType);
		m_curAgentController = Cast<ADeepDriveAgentControllerBase> (m_curAgent->GetController());

		m_curAgentIndex = index;
	}
}

void ADeepDriveSimulation::RegisterRandomStream(const FName &RandomStreamId, bool ReseedOnReset)
{
	if (RandomStreams.Contains(RandomStreamId) == false)
	{
		UDeepDriveRandomStream *randomStream = NewObject<UDeepDriveRandomStream>();
		if (randomStream)
		{
			randomStream->initialize(Seed);
			RandomStreams.Add(RandomStreamId, FDeepDriveRandomStreamData(randomStream, ReseedOnReset));
		}
	}
}

UDeepDriveRandomStream* ADeepDriveSimulation::GetRandomStream(const FName &RandomStreamId)
{
	return RandomStreams.Contains(RandomStreamId) ? RandomStreams[RandomStreamId].getRandomStream() : 0;
}

void ADeepDriveSimulation::OnDebugTrigger()
{
	if (m_curAgent)
	{
		m_curAgent->OnDebugTrigger();
	}
}

