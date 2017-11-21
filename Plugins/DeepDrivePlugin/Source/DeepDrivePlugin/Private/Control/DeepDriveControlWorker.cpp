
#include "DeepDrivePluginPrivatePCH.h"

#include "Public/Capture/CaptureDefines.h"
#include "Private/Control/DeepDriveControlWorker.h"
#include "Public/SharedMemory/SharedMemory.h"
#include "Public/Messages/DeepDriveControlMessages.h"

DEFINE_LOG_CATEGORY(LogDeepDriveControlWorker);

DeepDriveControlWorker::DeepDriveControlWorker(const FString &sharedMemName, uint32 sharedMemSize)
	:	m_WorkerThread(0)
	,	m_isStopped(false)
	,	m_SharedMemoryName(sharedMemName)
	,	m_SharedMemorySize(sharedMemSize)
	,	m_State(Idle)
	,	m_SharedMemory(0)
	,	m_lastMsgId(0)
{
	const FString name("DeepDriveControlWorker");
	m_WorkerThread = FRunnableThread::Create(this, *name, 0, TPri_Normal);

	m_SharedMemory = new SharedMemory;
}

DeepDriveControlWorker::~DeepDriveControlWorker()
{
}

bool DeepDriveControlWorker::Init()
{
	m_isStopped = false;
	m_State = WaitingForConnection;
	return true;
}

uint32 DeepDriveControlWorker::Run()
{
	do
	{
		if (!m_isStopped)
		{
			float sleepTime = 0.001;

			switch (m_State)
			{
				case Idle:
					break;

				case WaitingForConnection:
					if (connect())
						m_State = Connected;
					else
						sleepTime = 0.5f;
					break;

				case Connected:
					checkForMessage();
					break;
			}

			FPlatformProcess::Sleep(sleepTime);
		}

	} while (!m_isStopped);

	return 0;
}

void DeepDriveControlWorker::Stop()
{
	m_isStopped = true;
}

void DeepDriveControlWorker::Exit()
{
	delete this;
}

const DeepDriveMessageHeader* DeepDriveControlWorker::getMessage()
{
	const DeepDriveMessageHeader *msg = 0;
	m_MessageQueue.Dequeue(msg);
	return msg;
}

bool DeepDriveControlWorker::connect()
{
	bool connected = false;

	if (m_SharedMemory)
	{
		if (m_SharedMemory->tryConnect(m_SharedMemoryName, m_SharedMemorySize))
		{
			connected = true;
			UE_LOG(LogDeepDriveControlWorker, Log, TEXT("Successfully connected to %s with size %d"), *(m_SharedMemoryName), m_SharedMemorySize);
		}
	}

	return connected;
}

void DeepDriveControlWorker::checkForMessage()
{
	const DeepDriveMessageHeader *msg = reinterpret_cast<const DeepDriveMessageHeader*> (m_SharedMemory->lockForReading(-1));
	if (msg)
	{
		if	(	msg->message_id != 0
			&&	msg->message_id != m_lastMsgId
			)
		{
			m_MessageQueue.Enqueue(msg->clone());
		}

		m_lastMsgId = msg->message_id;
	}

	m_SharedMemory->unlock();
}


void DeepDriveControlWorker::handleControlMessage(const DeepDriveControlMessage &ctrlMsg)
{
	UE_LOG(LogDeepDriveControlWorker, Log, TEXT("Control message received: steering %f throttle %f brake %f handbrake %d"), ctrlMsg.steering, ctrlMsg.throttle, ctrlMsg.brake, ctrlMsg.handbrake);

}


void DeepDriveControlWorker::disconnect()
{
	if (m_SharedMemory)
	{
		m_SharedMemory->disconnect();
		m_State = Idle;
		UE_LOG(LogDeepDriveControlWorker, Log, TEXT("Successfully disconnected from %s"), *(m_SharedMemoryName));
	}
}
