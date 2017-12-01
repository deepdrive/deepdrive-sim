// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "Engine.h"
#include "Runtime/Core/Public/HAL/Runnable.h"

#include "Private/Server/DeepDriveMessageAssembler.h"

#include "Public/Server/Messages/DeepDriveMessageIds.h"

#include <map>
#include <functional>

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveClientConnection, Log, All);

namespace deepdrive { namespace server {
struct MessageHeader;
} }

class FSocket;

/**
 * 
 */
class DeepDriveClientConnection	:	public FRunnable
{
	typedef TQueue<deepdrive::server::MessageHeader*>	MessageQueue;

	typedef std::function< void(const deepdrive::server::MessageHeader&, bool) > HandleMessageFuncPtr;
	typedef std::map<deepdrive::server::MessageId, HandleMessageFuncPtr>	MessageHandlers;

public:

	DeepDriveClientConnection(FSocket *socket, uint32 clientId);

	~DeepDriveClientConnection();

	virtual bool Init();
	virtual uint32 Run();
	virtual void Stop();

	void terminate();

	void enqueueResponse(deepdrive::server::MessageHeader *message);

private:

	void shutdown();

	void handleClientRequest(const deepdrive::server::MessageHeader &message);

	void registerClient(const deepdrive::server::MessageHeader &message, bool isMaster);

	void unregisterClient(const deepdrive::server::MessageHeader &message, bool isMaster);

	bool resizeReceiveBuffer(uint32 minSize);

	FSocket								*m_Socket = 0;
	uint32								m_ClientId = 0;

	FRunnableThread						*m_WorkerThread = 0;

	MessageQueue						m_ResponseQueue;

	bool								m_isStopped = false;

	uint8								*m_ReceiveBuffer = 0;
	uint32								m_curReceiveBufferSize = 0;

	DeepDriveMessageAssembler			m_MessageAssembler;

	bool								m_isMaster = false;

	MessageHandlers						m_MessageHandlers;

};
