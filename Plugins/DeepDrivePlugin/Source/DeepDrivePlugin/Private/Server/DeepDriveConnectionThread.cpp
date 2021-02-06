// Fill out your copyright notice in the Description page of Project Settings.

#include "DeepDriveConnectionThread.h"

#include "Runtime/Sockets/Public/Sockets.h"
#include "Server/Messages/DeepDriveServerMessageHeader.h"
#include "Server/Messages/DeepDriveServerConnectionMessages.h"
#include "Server/Messages/DeepDriveServerConfigurationMessages.h"

#include "CaptureSink/SharedMemSink/SharedMemCaptureSinkComponent.h"
#include "Server/DeepDriveServer.h"

using namespace deepdrive::server;

DEFINE_LOG_CATEGORY(LogDeepDriveConnectionThread);

DeepDriveConnectionThread::DeepDriveConnectionThread(const FString &threadName)
	:	m_ThreadName(threadName)
	,	m_MessageAssembler()
{
	(void)resizeReceiveBuffer(64 * 1024);

}

DeepDriveConnectionThread::~DeepDriveConnectionThread()
{
	shutdown();
}

void DeepDriveConnectionThread::start()
{
	if(m_WorkerThread == 0)
	{
		m_WorkerThread = FRunnableThread::Create(this, *m_ThreadName, 0, TPri_Normal);
	}

}


void DeepDriveConnectionThread::Stop()
{
	m_isStopped = true;
}

void DeepDriveConnectionThread::Exit()
{
	delete this;
}

void DeepDriveConnectionThread::shutdown()
{
}

bool DeepDriveConnectionThread::resizeReceiveBuffer(uint32 minSize)
{
	if (minSize > m_curReceiveBufferSize)
	{
		minSize += minSize / 4;
		delete m_ReceiveBuffer;
		m_ReceiveBuffer = reinterpret_cast<uint8*> (FMemory::Malloc(minSize));
		m_curReceiveBufferSize = minSize;
	}
	return m_ReceiveBuffer != 0;
}

void DeepDriveConnectionThread::enqueueResponse(deepdrive::server::MessageHeader *message)
{
	if (message)
	{
		m_ResponseQueue.Enqueue(message);
	}
}

