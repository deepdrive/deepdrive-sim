#pragma once

#include "Engine.h"
#include "Runtime/Core/Public/HAL/Runnable.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveControlWorker, Log, All);

class SharedMemory;
struct DeepDriveControlMessage;

class DeepDriveControlWorker	:	public FRunnable
{
	enum State
	{
		Idle,
		WaitingForConnection,
		Connected
	};

public:

	DeepDriveControlWorker(const FString &sharedMemName, uint32 sharedMemSize);
	virtual ~DeepDriveControlWorker();

	virtual bool Init();
	virtual uint32 Run();
	virtual void Stop();

	virtual void Exit();

	const DeepDriveMessageHeader* getMessage();

private:

	bool connect();
	void checkForMessage();

	void handleControlMessage(const DeepDriveControlMessage &ctrlMsg);

	void disconnect();

	FRunnableThread								*m_WorkerThread;
	bool										m_isStopped;

	FString										m_SharedMemoryName;
	uint32										m_SharedMemorySize;

	State										m_State;

	SharedMemory								*m_SharedMemory;
	uint32										m_lastMsgId;


	TQueue<const DeepDriveMessageHeader*>		m_MessageQueue;
};
