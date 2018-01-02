// Fill out your copyright notice in the Description page of Project Settings.

#include "DeepDrivePluginPrivatePCH.h"
#include "Private/Server/DeepDriveServer.h"
#include "Private/Server/DeepDriveConnectionListener.h"
#include "Private/Server/DeepDriveClientConnection.h"

#include "Public/Server/DeepDriveServerProxy.h"
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

	m_MessageHandlers[deepdrive::server::MessageId::RequestAgentControlRequest] = std::bind(&DeepDriveServer::handleRequestAgentControl, this, std::placeholders::_1);
	m_MessageHandlers[deepdrive::server::MessageId::ReleaseAgentControlRequest] = std::bind(&DeepDriveServer::handleReleaseAgentControl, this, std::placeholders::_1);
	m_MessageHandlers[deepdrive::server::MessageId::SetAgentControlValuesRequest] = std::bind(&DeepDriveServer::setAgentControlValues, this, std::placeholders::_1);
	m_MessageHandlers[deepdrive::server::MessageId::ResetAgentRequest] = std::bind(&DeepDriveServer::resetAgent, this, std::placeholders::_1);
}

DeepDriveServer::~DeepDriveServer()
{
}



bool DeepDriveServer::RegisterProxy(ADeepDriveServerProxy &proxy)
{
	bool registered = false;

	bool isValid = false;
	int32 ipAddress[4];

	TArray<FString> parts;
	if(proxy.IPAddress.ParseIntoArray(parts, TEXT("."), 1) == 4)
	{
		ipAddress[0] = FCString::Atoi(*parts[0]);
		ipAddress[1] = FCString::Atoi(*parts[1]);
		ipAddress[2] = FCString::Atoi(*parts[2]);
		ipAddress[3] = FCString::Atoi(*parts[3]);

		if	(	ipAddress[0] >= 0 && ipAddress[0] <= 255
			&&	ipAddress[1] >= 0 && ipAddress[1] <= 255
			&&	ipAddress[2] >= 0 && ipAddress[2] <= 255
			&&	ipAddress[3] >= 0 && ipAddress[3] <= 255
			)
		{
			isValid = true;
		}

	}
		
	if	(	isValid
		&&	proxy.Port >= 1 && proxy.Port <= 65535
		)
	{
		m_Proxy = &proxy;
		m_ConnectionListener = new DeepDriveConnectionListener(ipAddress[0], ipAddress[1], ipAddress[2], ipAddress[3], proxy.Port);

		registered = true;
	}

	return registered;
}

void DeepDriveServer::UnregisterProxy(ADeepDriveServerProxy &proxy)
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
			m_MasterClientId = clientId;
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
	SIncomingConnection *incoming = 0;
	if	(m_IncomingConnections.Dequeue(incoming)
		&&	incoming != 0
		)
	{
		UE_LOG(LogDeepDriveServer, Log, TEXT("Incoming client connection asdf from %s"), *(incoming->remote_address->ToString(true)));
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
	const deepdrive::server::RegisterCaptureCameraRequest &req = static_cast<const deepdrive::server::RegisterCaptureCameraRequest&> (message);
	DeepDriveClientConnection *client = m_Clients.Find(req.client_id)->connection;
	if (client)
	{
		FVector relPos(req.relative_position[0], req.relative_position[1], req.relative_position[2]);
		FVector relRot(req.relative_rotation[0], req.relative_rotation[1], req.relative_rotation[2]);
		int32 cameraId = m_Proxy->RegisterCamera(req.horizontal_field_of_view, req.capture_width, req.capture_height, relPos, relRot);

		UE_LOG(LogDeepDriveServer, Log, TEXT("Camera registered %d %d"), req.client_id, cameraId);
		client->enqueueResponse(new deepdrive::server::RegisterCaptureCameraResponse(cameraId));
	}
}

void DeepDriveServer::handleRequestAgentControl(const deepdrive::server::MessageHeader &message)
{
	const deepdrive::server::RequestAgentControlRequest &req = static_cast<const deepdrive::server::RequestAgentControlRequest&> (message);
	DeepDriveClientConnection *client = m_Clients.Find(req.client_id)->connection;
	if (client)
	{
		bool ctrlGranted = m_Proxy->RequestAgentControl();
		UE_LOG(LogDeepDriveServer, Log, TEXT("Control over agent granted %d %c"), req.client_id, ctrlGranted ? 'T' : 'F');
		client->enqueueResponse(new deepdrive::server::RequestAgentControlResponse(ctrlGranted));
	}
}

void DeepDriveServer::handleReleaseAgentControl(const deepdrive::server::MessageHeader &message)
{
	const deepdrive::server::ReleaseAgentControlRequest &req = static_cast<const deepdrive::server::ReleaseAgentControlRequest&> (message);
	DeepDriveClientConnection *client = m_Clients.Find(req.client_id)->connection;
	if (client)
	{
		m_Proxy->ReleaseAgentControl();
		client->enqueueResponse(new deepdrive::server::ReleaseAgentControlResponse(true));
	}
	else
	{
		UE_LOG(LogDeepDriveServer, Log, TEXT("Ignoring release control request, no clients connected"));
	}
}

void DeepDriveServer::resetAgent(const deepdrive::server::MessageHeader &message)
{
	const deepdrive::server::ResetAgentRequest &req = static_cast<const deepdrive::server::ResetAgentRequest&> (message);
	DeepDriveClientConnection *client = m_Clients.Find(req.client_id)->connection;
	if (client)
	{
		m_Proxy->ResetAgent();
		UE_LOG(LogDeepDriveServer, Log, TEXT("Agent reset %d"), req.client_id);
	}
	else{
		UE_LOG(LogDeepDriveServer, Log, TEXT("No client, ignoring reset"));
	}
}

void DeepDriveServer::onAgentReset(bool success)
{
	if (m_MasterClientId)
	{
		DeepDriveClientConnection *client = m_Clients.Find(m_MasterClientId)->connection;
		if (client && client->isMaster())
		{
			client->enqueueResponse(new deepdrive::server::ResetAgentResponse(success));
			UE_LOG(LogDeepDriveServer, Log, TEXT("[%d] Agent reset success %c"), m_MasterClientId, success ? 'T' : 'F');
		}
		else
		{
			UE_LOG(LogDeepDriveServer, Log, TEXT("onResetAgent: No master client found for %d"), m_MasterClientId);
		}
	}
}


void DeepDriveServer::setAgentControlValues(const deepdrive::server::MessageHeader &message)
{
	const deepdrive::server::SetAgentControlValuesRequest &req = static_cast<const deepdrive::server::SetAgentControlValuesRequest&> (message);
	m_Proxy->SetAgentControlValues(req.steering, req.throttle, req.brake, req.handbrake != 0 ? true : false);
	// UE_LOG(LogDeepDriveServer, Log, TEXT("Control values received from %d"), req.client_id);
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
		