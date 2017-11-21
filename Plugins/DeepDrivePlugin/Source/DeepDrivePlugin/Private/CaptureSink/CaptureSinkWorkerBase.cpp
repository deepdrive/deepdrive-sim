
#include "DeepDrivePluginPrivatePCH.h"

#include "Public/Capture/CaptureDefines.h"
#include "Private/CaptureSink/CaptureSinkWorkerBase.h"


CaptureSinkWorkerBase::CaptureSinkWorkerBase(const FString &name)
{
	m_Semaphore = FGenericPlatformProcess::GetSynchEventFromPool(false);
	m_WorkerThread = FRunnableThread::Create(this, *(name) , 0, TPri_AboveNormal);
}

CaptureSinkWorkerBase::~CaptureSinkWorkerBase()
{
	if (m_Semaphore)
		FGenericPlatformProcess::ReturnSynchEventToPool(m_Semaphore);

	delete m_WorkerThread;
}

bool CaptureSinkWorkerBase::Init()
{
	m_isStopped = false;
	return true;
}

uint32 CaptureSinkWorkerBase::Run()
{
	do
	{
		(void) m_Semaphore->Wait();
		if(!m_isStopped)
		{
			SCaptureSinkJobData *jobData = 0;

			while	(	m_JobDataQueue.Dequeue(jobData)
					&&	jobData
					)
			{
				const bool continueExecuting = execute(*jobData);

				delete jobData;

				if(continueExecuting)
					FPlatformProcess::Sleep(0.001);
				else
					break;
			}
		}

	} while (!m_isStopped);

	return 0;
}

void CaptureSinkWorkerBase::Stop()
{
	m_isStopped = true;
	m_Semaphore->Trigger();
}


void CaptureSinkWorkerBase::process(SCaptureSinkJobData &jobData)
{
	m_JobDataQueue.Enqueue(&jobData);
	m_Semaphore->Trigger();
}

bool CaptureSinkWorkerBase::execute(SCaptureSinkJobData &jobData)
{
	return false;
}
