// Fill out your copyright notice in the Description page of Project Settings.

#include "DeepDrivePluginPrivatePCH.h"
#include "Private/Server/DeepDriveServer.h"
#include "Private/Server/DeepDriveConnectionListener.h"
#include "Private/Server/DeepDriveClientConnection.h"
#include "Private/Capture/DeepDriveCapture.h"

#include "Public/Server/IDeepDriveServerProxy.h"
#include "Public/Server/Messages/DeepDriveServerConfigurationMessages.h"
#include "Public/Server/Messages/DeepDriveServerControlMessages.h"

#include "Runtime/Networking/Public/Interfaces/IPv4/IPv4SubnetMask.h"
#include "Runtime/Networking/Public/Interfaces/IPv4/IPv4Address.h"
#include "Runtime/Sockets/Public/IPAddress.h"

DEFINE_LOG_CATEGORY(LogDeepDriveServer);

DeepDriveServer* DeepDriveServer::theInstance = 0;

DeepDriveServer& DeepDriveServer::GetInstance()
{
	if(theInstance == 0)
	{
		theInstance = new DeepDriveServer;
	}

	return *theInstance;
}

void DeepDriveServer::Destroy()
{
	delete theInstance;
	theInstance = 0;
}


DeepDriveServer::DeepDriveServer()
{
	UE_LOG(LogDeepDriveServer, Log, TEXT("DeepDriveServer created") );

	m_MessageHandlers[deepdrive::server::MessageId::RegisterCaptureCameraRequest] = std::bind(&DeepDriveServer::handleRegisterCamera, this, std::placeholders::_1);
	m_MessageHandlers[deepdrive::server::MessageId::UnregisterCaptureCameraRequest] = std::bind(&DeepDriveServer::handleUnregisterCamera, this, std::placeholders::_1);

	m_MessageHandlers[deepdrive::server::MessageId::RequestAgentControlRequest] = std::bind(&DeepDriveServer::handleRequestAgentControl, this, std::placeholders::_1);
	m_MessageHandlers[deepdrive::server::MessageId::ReleaseAgentControlRequest] = std::bind(&DeepDriveServer::handleReleaseAgentControl, this, std::placeholders::_1);
	m_MessageHandlers[deepdrive::server::MessageId::SetAgentControlValuesRequest] = std::bind(&DeepDriveServer::setAgentControlValues, this, std::placeholders::_1);
	m_MessageHandlers[deepdrive::server::MessageId::ResetAgentRequest] = std::bind(&DeepDriveServer::resetAgent, this, std::placeholders::_1);

	m_MessageHandlers[deepdrive::server::MessageId::ActivateSynchronousSteppingRequest] = std::bind(&DeepDriveServer::activateSynchronousStepping, this, std::placeholders::_1);
	m_MessageHandlers[deepdrive::server::MessageId::DeactivateSynchronousSteppingRequest] = std::bind(&DeepDriveServer::deactivateSynchronousStepping, this, std::placeholders::_1);
	m_MessageHandlers[deepdrive::server::MessageId::AdvanceSynchronousSteppingRequest] = std::bind(&DeepDriveServer::advanceSynchronousStepping, this, std::placeholders::_1);

	m_MessageHandlers[deepdrive::server::MessageId::SetViewModeRequest] = std::bind(&DeepDriveServer::setViewMode, this, std::placeholders::_1);

}

DeepDriveServer::~DeepDriveServer()
{
}

bool DeepDriveServer::RegisterProxy(IDeepDriveServerProxy &proxy, const FString &clientIpAddress, uint16 clientPort)
{
	bool registered = false;

	int32 clientIpParts[4];

	if(convertIpAddress(clientIpAddress, clientIpParts))
	{
		m_Proxy = &proxy;
		m_ConnectionListener = new DeepDriveConnectionListener(clientIpParts[0], clientIpParts[1], clientIpParts[2], clientIpParts[3], clientPort);

		m_MessageQueue.Empty();
		m_SteppingClient = 0;
		m_State = Continous;

		registered = true;
	}

	return registered;
}

void DeepDriveServer::UnregisterProxy(IDeepDriveServerProxy &proxy)
{
	if (m_Proxy == &proxy)
	{
		if (m_ConnectionListener)
		{
			m_ConnectionListener->terminate();
		}

		for (auto &clientData : m_Clients)
		{
			if (clientData.Value.connection)
				clientData.Value.connection->Stop();
		}
		m_Clients.Empty();
		m_MasterClientId = 0;
	}
}

uint32 DeepDriveServer::registerClient(DeepDriveClientConnection *client, bool &isMaster)
{
	FScopeLock lock(&m_ClientMutex);
	const uint32 clientId = m_nextClientId++;
	m_Clients.Add(clientId, SClient(clientId, client));

	if (isMaster)
	{
		if (m_MasterClientId == 0)
		{
			m_MasterClientId = clientId;
		}
		else
			isMaster = false;
	}

	if (m_Proxy)
		m_Proxy->RegisterClient(clientId, isMaster);

	return clientId;
}

void DeepDriveServer::unregisterClient(uint32 clientId)
{
	if (m_Proxy)
		m_Proxy->UnregisterClient(clientId, m_MasterClientId == clientId);

	FScopeLock lock(&m_ClientMutex);

	if (m_Clients.Find(clientId))
	{
		DeepDriveClientConnection *client = m_Clients[clientId].connection;
		m_Clients.Remove(clientId);
		client->Stop();

		if (m_MasterClientId == clientId)
		{
			m_MasterClientId = 0;
		}
	}
}

void DeepDriveServer::update(float DeltaSeconds)
{
	switch (m_State)
	{
		case Continous:
		case Stepping_Idle:
			handleMessageQueues();
			break;

		case Stepping_Advance:
			if	(	FPlatformTime::Seconds() >= m_AdvanceEndTime
				&&	m_World
				)
			{

				UGameplayStatics::SetGamePaused(m_World, true);
				m_State = Stepping_WaitForCapture;
				CaptureFinishedDelegate captureFinishedDelegate;
				captureFinishedDelegate.BindRaw(this, &DeepDriveServer::onCaptureFinished);
				DeepDriveCapture::GetInstance().onNextCapture(captureFinishedDelegate);
			}
			break;

		case Stepping_WaitForCapture:
			break;
	}
}

void DeepDriveServer::handleMessageQueues()
{
	SIncomingConnection *incoming = 0;
	if (m_IncomingConnections.Dequeue(incoming)
		&& incoming != 0
		)
	{
		UE_LOG(LogDeepDriveServer, Log, TEXT("Incoming client connection from %s"), *(incoming->remote_address->ToString(true)));
		DeepDriveClientConnection *client = new DeepDriveClientConnection(incoming->socket);
	}

	deepdrive::server::MessageHeader *message = 0;
	if	(	m_MessageQueue.Dequeue(message)
		&&	message
		)
	{
		handleMessage(*message);
		FMemory::Free(message);
	}
}

void DeepDriveServer::handleMessage(const deepdrive::server::MessageHeader &message)
{
	if (m_Proxy)
	{
		MessageHandlers::iterator fIt = m_MessageHandlers.find(message.message_id);

		if (fIt != m_MessageHandlers.end())
			fIt->second(message);
	}
}

void DeepDriveServer::handleRegisterCamera(const deepdrive::server::MessageHeader &message)
{
	if(m_Clients.Num() > 0)
	{
		const deepdrive::server::RegisterCaptureCameraRequest &req = static_cast<const deepdrive::server::RegisterCaptureCameraRequest&> (message);
		DeepDriveClientConnection *client = m_Clients.Find(req.client_id)->connection;
		if (client)
		{
			int32 cameraId = 0;
			if (m_Proxy && client->isMaster())
			{
				FVector relPos(req.relative_position[0], req.relative_position[1], req.relative_position[2]);
				FVector relRot(req.relative_rotation[0], req.relative_rotation[1], req.relative_rotation[2]);
				cameraId = m_Proxy->RegisterCaptureCamera(req.horizontal_field_of_view, req.capture_width, req.capture_height, relPos, relRot, req.camera_label);

				UE_LOG(LogDeepDriveServer, Log, TEXT("Camera registered %d %d"), req.client_id, cameraId);
			}
			else
			{
				UE_LOG(LogDeepDriveServer, Log, TEXT("Client %d isn't master or no proxy, registering camera not allowed"), req.client_id);
			}

			client->enqueueResponse(new deepdrive::server::RegisterCaptureCameraResponse(cameraId));
		}
	}
}

void DeepDriveServer::handleUnregisterCamera(const deepdrive::server::MessageHeader &message)
{
	if(m_Clients.Num() > 0)
	{
		const deepdrive::server::UnregisterCaptureCameraRequest &req = static_cast<const deepdrive::server::UnregisterCaptureCameraRequest&> (message);
		DeepDriveClientConnection *client = m_Clients.Find(req.client_id)->connection;
		if (client)
		{
			if (m_Proxy)
			{
				m_Proxy->UnregisterCaptureCamera(req.camera_id);
				client->enqueueResponse(new deepdrive::server::UnregisterCaptureCameraResponse(true));

				UE_LOG(LogDeepDriveServer, Log, TEXT("Camera unregistered %d %d"), req.client_id, req.camera_id);
			}
			else
			{
				client->enqueueResponse(new deepdrive::server::UnregisterCaptureCameraResponse(false));
				UE_LOG(LogDeepDriveServer, Log, TEXT("No proxy, unregistering camera not possible"));
			}

		}
		else
			UE_LOG(LogDeepDriveServer, Error, TEXT("Client %d not found"), req.client_id);
	}
}


void DeepDriveServer::handleRequestAgentControl(const deepdrive::server::MessageHeader &message)
{
	if(m_Clients.Num() > 0)
	{
		const deepdrive::server::RequestAgentControlRequest &req = static_cast<const deepdrive::server::RequestAgentControlRequest&> (message);
		DeepDriveClientConnection *client = m_Clients.Find(req.client_id)->connection;
		if (client)
		{
			bool ctrlGranted = false;
			if (m_Proxy && client->isMaster())
			{
				ctrlGranted = m_Proxy->RequestAgentControl();
				UE_LOG(LogDeepDriveServer, Log, TEXT("Control over agent granted %d %c"), req.client_id, ctrlGranted ? 'T' : 'F');
			}
			else
			{
				UE_LOG(LogDeepDriveServer, Log, TEXT("Client %d isn't master or no proxy, control not granted"), req.client_id);
			}

			client->enqueueResponse(new deepdrive::server::RequestAgentControlResponse(ctrlGranted));
		}
	}
}

void DeepDriveServer::handleReleaseAgentControl(const deepdrive::server::MessageHeader &message)
{
	if (m_Clients.Num() > 0)
	{
		const deepdrive::server::ReleaseAgentControlRequest &req = static_cast<const deepdrive::server::ReleaseAgentControlRequest&> (message);
		DeepDriveClientConnection *client = m_Clients.Find(req.client_id)->connection;
		if (client)
		{
			if (m_Proxy && client->isMaster())
			{
				m_Proxy->ReleaseAgentControl();
			}
			else
				UE_LOG(LogDeepDriveServer, Log, TEXT("Client %d isn't master or no proxy, control not released"), req.client_id);

			client->enqueueResponse(new deepdrive::server::ReleaseAgentControlResponse(true));
		}
		else
		{
			UE_LOG(LogDeepDriveServer, Log, TEXT("Ignoring release control request, no clients connected"));
		}
	}
}

void DeepDriveServer::resetAgent(const deepdrive::server::MessageHeader &message)
{
	UE_LOG(LogDeepDriveServer, Log, TEXT("Inside reset agent"));

	if(m_Clients.Num() > 0)
	{
		UE_LOG(LogDeepDriveServer, Log, TEXT("Reset agent with client num %d"), m_Clients.Num());


		const deepdrive::server::ResetAgentRequest &req = static_cast<const deepdrive::server::ResetAgentRequest&> (message);
		DeepDriveClientConnection *client = m_Clients.Find(req.client_id)->connection;
		if (client)
		{
			if (m_Proxy && client->isMaster())
			{
				m_Proxy->ResetAgent();
				UE_LOG(LogDeepDriveServer, Log, TEXT("Agent reset %d"), req.client_id);
			}
			else
			{
				UE_LOG(LogDeepDriveServer, Log, TEXT("Agent reset %d"), req.client_id);
				client->enqueueResponse(new deepdrive::server::ResetAgentResponse(false));
			}
		}
		else
		{
			UE_LOG(LogDeepDriveServer, Log, TEXT("No client, ignoring reset"));
		}
	}
	else
	{
		UE_LOG(LogDeepDriveServer, Log, TEXT("No clients, ignoring reset"));
	}
}

void DeepDriveServer::onAgentReset(bool success)
{
	if (m_Clients.Num() > 0)
	{
		DeepDriveClientConnection *client = m_Clients.Find(m_MasterClientId)->connection;
		if (client)
		{
			client->enqueueResponse(new deepdrive::server::ResetAgentResponse(success));
			UE_LOG(LogDeepDriveServer, Log, TEXT("[%d] Agent reset success %c"), m_MasterClientId, success ? 'T' : 'F');
		}
		else
		{
			UE_LOG(LogDeepDriveServer, Log, TEXT("onResetAgent: No master client found for %d"), m_MasterClientId);
		}
	}
	else
	{
		UE_LOG(LogDeepDriveServer, Log, TEXT("No clients , ignoring reset"));
	}
}

void DeepDriveServer::setAgentControlValues(const deepdrive::server::MessageHeader &message)
{
	if (m_Proxy && m_Clients.Num() > 0)
	{
		const deepdrive::server::SetAgentControlValuesRequest &req = static_cast<const deepdrive::server::SetAgentControlValuesRequest&> (message);
		DeepDriveClientConnection *client = m_Clients.Find(req.client_id)->connection;
		if (client && client->isMaster())
		{
			m_Proxy->SetAgentControlValues(req.steering, req.throttle, req.brake, req.handbrake != 0 ? true : false);
			// UE_LOG(LogDeepDriveServer, Log, TEXT("Control values received from %d"), req.client_id);
		}
	}
}

void DeepDriveServer::activateSynchronousStepping(const deepdrive::server::MessageHeader &message)
{
	if (m_Clients.Num() > 0)
	{
		const deepdrive::server::ActivateSynchronousSteppingRequest &req = static_cast<const deepdrive::server::ActivateSynchronousSteppingRequest&> (message);
		DeepDriveClientConnection *client = m_Clients.Find(req.client_id)->connection;
		if (client)
		{
			if	(	client->isMaster() && m_State == Continous
				&&	m_World
				)
			{
				m_State = Stepping_Idle;
				UGameplayStatics::SetGamePaused(m_World, true);
				UE_LOG(LogDeepDriveServer, Log, TEXT("SynchronousStepping activated"));


				client->enqueueResponse(new deepdrive::server::ActivateSynchronousSteppingResponse(true));
			}
			else
				client->enqueueResponse(new deepdrive::server::ActivateSynchronousSteppingResponse(false));
		}
	}
}

void DeepDriveServer::deactivateSynchronousStepping(const deepdrive::server::MessageHeader &message)
{
	if (m_Clients.Num() > 0)
	{
		const deepdrive::server::DeactivateSynchronousSteppingRequest &req = static_cast<const deepdrive::server::DeactivateSynchronousSteppingRequest&> (message);
		DeepDriveClientConnection *client = m_Clients.Find(req.client_id)->connection;
		if (client)
		{
			if	(	client->isMaster() && m_State == Stepping_Idle
				&&	m_World
				)
			{
				m_State = Continous;
				UGameplayStatics::SetGamePaused(m_World, false);
				UE_LOG(LogDeepDriveServer, Log, TEXT("SynchronousStepping deactivated"));

				client->enqueueResponse(new deepdrive::server::DeactivateSynchronousSteppingResponse(true));
			}
			else
				client->enqueueResponse(new deepdrive::server::DeactivateSynchronousSteppingResponse(false));
		}
	}
}

void DeepDriveServer::advanceSynchronousStepping(const deepdrive::server::MessageHeader &message)
{
	if (m_Clients.Num() > 0)
	{
		const deepdrive::server::AdvanceSynchronousSteppingRequest &req = static_cast<const deepdrive::server::AdvanceSynchronousSteppingRequest&> (message);
		DeepDriveClientConnection *client = m_Clients.Find(req.client_id)->connection;
		if (client)
		{
			if(m_Proxy && client->isMaster() && m_State == Stepping_Idle)
			{
				m_SteppingClient = client;
				UGameplayStatics::SetGamePaused(m_World, false);
				m_Proxy->SetAgentControlValues(req.steering, req.throttle, req.brake, req.handbrake != 0 ? true : false);

				m_State = Stepping_Advance;
				m_AdvanceEndTime = FPlatformTime::Seconds() + req.time_step;
			}
			else
				client->enqueueResponse(new deepdrive::server::AdvanceSynchronousSteppingResponse(-1));
		}
	}
}

void DeepDriveServer::setViewMode(const deepdrive::server::MessageHeader &message)
{
	if (m_Clients.Num() > 0)
	{
		const deepdrive::server::SetViewModeRequest &req = static_cast<const deepdrive::server::SetViewModeRequest&> (message);

		FString viewMode(ANSI_TO_TCHAR(req.view_mode));

		DeepDriveClientConnection *client = m_Clients.Find(req.client_id)->connection;
		if (client)
		{
			if (m_Proxy && client->isMaster())
			{
				const bool res = m_Proxy->SetViewMode(req.camera_id, viewMode);
				client->enqueueResponse(new deepdrive::server::SetViewModeResponse(res));
			}
			else
				client->enqueueResponse(new deepdrive::server::SetViewModeResponse(false));
		}
	}
}

void DeepDriveServer::onCaptureFinished(int32 seqNr)
{
	UE_LOG(LogDeepDriveServer, Verbose, TEXT("DeepDriveServer::onCaptureFinished %d"), seqNr);

	if (m_SteppingClient)
		m_SteppingClient->enqueueResponse(new deepdrive::server::AdvanceSynchronousSteppingResponse(seqNr));
	m_SteppingClient = 0;
	m_State = Stepping_Idle;
}

void DeepDriveServer::addIncomingConnection(FSocket *socket, TSharedRef<FInternetAddr> remoteAddr)
{
	m_IncomingConnections.Enqueue(new SIncomingConnection(socket, remoteAddr));
}


void DeepDriveServer::initializeClient(uint32 clientId)
{
}

void  DeepDriveServer::enqueueMessage(deepdrive::server::MessageHeader *message)
{
	if(message)
		m_MessageQueue.Enqueue(message);
}

bool DeepDriveServer::convertIpAddress(const FString &ipAddress, int32 *ipParts)
{
	bool isValid = false;

	TArray<FString> parts;
	if (ipAddress.ParseIntoArray(parts, TEXT("."), 1) == 4)
	{
		ipParts[0] = FCString::Atoi(*parts[0]);
		ipParts[1] = FCString::Atoi(*parts[1]);
		ipParts[2] = FCString::Atoi(*parts[2]);
		ipParts[3] = FCString::Atoi(*parts[3]);

		if (	ipParts[0] >= 0 && ipParts[0] <= 255
			&&	ipParts[1] >= 0 && ipParts[1] <= 255
			&&	ipParts[2] >= 0 && ipParts[2] <= 255
			&&	ipParts[3] >= 0 && ipParts[3] <= 255
			)
		{
			isValid = true;
		}

	}
	return isValid;
}

#if 0

void DeepDriveSimulationServerProxy::activateSynchronousStepping()
{
	SetTickableWhenPaused(true);
	UGameplayStatics::SetGamePaused(GetWorld(), true);
	UE_LOG(LogDeepDriveSimulationServerProxy, Log, TEXT("SynchronousStepping activated"));
}

void DeepDriveSimulationServerProxy::deactivateSynchronousStepping()
{
	UGameplayStatics::SetGamePaused(GetWorld(), false);
	SetTickableWhenPaused(false);
	UE_LOG(LogDeepDriveSimulationServerProxy, Log, TEXT("SynchronousStepping deactivated"));
}

void DeepDriveSimulationServerProxy::advanceSynchronousStepping(float steering, float throttle, float brake, bool handbrake)
{
	UGameplayStatics::SetGamePaused(GetWorld(), false);
	SetAgentControlValues(steering, throttle, brake, handbrake != 0 ? true : false);
}

#endif
