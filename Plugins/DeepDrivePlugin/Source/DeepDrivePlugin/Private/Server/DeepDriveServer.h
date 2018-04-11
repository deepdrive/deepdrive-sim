// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "Engine.h"

#include "Public/Server/Messages/DeepDriveMessageIds.h"

#include <map>
#include <functional>

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveServer, Log, All);

class DeepDriveConnectionListener;
class DeepDriveClientConnection;
class ADeepDriveServerProxy;
class IDeepDriveServerProxy;

class FSocket;

namespace deepdrive { namespace server {
struct MessageHeader;
} }

/**
 * 
 */
class DeepDriveServer
{
	struct SIncomingConnection
	{
		SIncomingConnection(FSocket *s, const TSharedRef<FInternetAddr> &r)
			: socket(s)
			, remote_address(r)
		{	}

		FSocket							*socket;
		TSharedRef<FInternetAddr>		remote_address;
	};

	struct SClient
	{
		SClient(uint32 cId = 0, DeepDriveClientConnection *c = 0)
			:	client_id(cId)
			,	connection(c)
		{	}

		uint32			client_id;
		DeepDriveClientConnection		*connection;
	};

	typedef TQueue<deepdrive::server::MessageHeader*> MessageQueue;

	typedef std::function< void(const deepdrive::server::MessageHeader&) > HandleMessageFuncPtr;
	typedef std::map<deepdrive::server::MessageId, HandleMessageFuncPtr>	MessageHandlers;


public:

	static DeepDriveServer& GetInstance();

	static void Destroy();

	bool RegisterProxy(IDeepDriveServerProxy &proxy);

	void UnregisterProxy(IDeepDriveServerProxy &proxy);

	uint32 registerClient(DeepDriveClientConnection *client, bool &isMaster);

	void unregisterClient(uint32 clientId);

	void update(float DeltaSeconds);

	void addIncomingConnection(FSocket *socket, TSharedRef<FInternetAddr> remoteAddr);

	void enqueueMessage(deepdrive::server::MessageHeader *message);

	void onAgentReset(bool success);

private:

	DeepDriveServer();
	~DeepDriveServer();

	void initializeClient(uint32 clientId);

	void handleMessage(const deepdrive::server::MessageHeader &message);

	void handleRegisterCamera(const deepdrive::server::MessageHeader &message);

	void handleRequestAgentControl(const deepdrive::server::MessageHeader &message);
	void handleReleaseAgentControl(const deepdrive::server::MessageHeader &message);
	void resetAgent(const deepdrive::server::MessageHeader &message);

	void setAgentControlValues(const deepdrive::server::MessageHeader &message);

	DeepDriveConnectionListener		*m_ConnectionListener = 0;

	IDeepDriveServerProxy			*m_Proxy = 0;

	TQueue<SIncomingConnection*>	m_IncomingConnections;

	MessageQueue					m_MessageQueue;
	MessageHandlers					m_MessageHandlers;

	TMap<uint32, SClient>			m_Clients;
	uint32							m_nextClientId = 1;
	uint32							m_MasterClientId = 0;
	FCriticalSection				m_ClientMutex;

	TArray<DeepDriveClientConnection*>	m_ClientConnections;

	static DeepDriveServer			*theInstance;
};
