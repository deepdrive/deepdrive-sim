// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "Engine.h"
#include "Runtime/Core/Public/HAL/Runnable.h"

#include "Private/Server/DeepDriveMessageAssembler.h"

#include "Public/Server/Messages/DeepDriveMessageIds.h"

#include <map>
#include <functional>

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveConnectionThread, Log, All);

namespace deepdrive { namespace server {
struct MessageHeader;
} }

class FSocket;

/**
 * 
 */
class DeepDriveConnectionThread	:	public FRunnable
{
	typedef TQueue<deepdrive::server::MessageHeader*, EQueueMode::Mpsc>	MessageQueue;

	typedef std::function< void(const deepdrive::server::MessageHeader&, bool) > HandleMessageFuncPtr;
	typedef std::map<deepdrive::server::MessageId, HandleMessageFuncPtr>	MessageHandlers;

public:

	DeepDriveConnectionThread(const FString &threadName);

	~DeepDriveConnectionThread();

	void start();

	virtual void Stop();
	virtual void Exit();

	void enqueueResponse(deepdrive::server::MessageHeader *message);

protected:

	virtual void shutdown();

	bool resizeReceiveBuffer(uint32 minSize);

	FString								m_ThreadName;
	FRunnableThread						*m_WorkerThread = 0;

	MessageQueue						m_ResponseQueue;

	bool								m_isStopped = false;

	uint8								*m_ReceiveBuffer = 0;
	uint32								m_curReceiveBufferSize = 0;

	DeepDriveMessageAssembler			m_MessageAssembler;

};

