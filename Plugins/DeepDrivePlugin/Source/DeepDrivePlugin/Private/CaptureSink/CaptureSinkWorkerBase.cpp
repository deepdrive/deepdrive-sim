
#include "DeepDrivePluginPrivatePCH.h"

#include "Public/Capture/CaptureDefines.h"
#include "Private/CaptureSink/CaptureSinkWorkerBase.h"
#include "Private/Capture/CaptureBuffer.h"


CaptureSinkWorkerBase::CaptureSinkWorkerBase(const FString &name)
{
	m_Semaphore = FGenericPlatformProcess::GetSynchEventFromPool(false);
	m_WorkerThread = FRunnableThread::Create(this, *(name) , 0, TPri_Highest);
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

				const uint32 curJobCtr = m_JobCounter.Decrement();

				for(auto &data : jobData->captures)
				{
					if(data.capture_buffer)
						data.capture_buffer->release();
				}

	

				delete jobData;
/*
				if(continueExecuting || curJobCtr > 0)
					FPlatformProcess::Sleep(0.001);
				else
					break;
*/
			}
		}

	} while (!m_isStopped);

	return 0;
}

void CaptureSinkWorkerBase::kill()
{
	m_WorkerThread->Kill(true);
}

void CaptureSinkWorkerBase::Stop()
{
	m_isStopped = true;
	m_Semaphore->Trigger();
}


void CaptureSinkWorkerBase::process(SCaptureSinkJobData &jobData)
{
	m_JobDataQueue.Enqueue(&jobData);

	uint32 curJobCount = m_JobCounter.Increment();
	while(curJobCount > 2)
	{
		SCaptureSinkJobData *jd = 0;
		if (m_JobDataQueue.Dequeue(jd))
		{
			if (jd)
			{
				for (auto &c : jd->captures)
					if (c.capture_buffer)
						c.capture_buffer->release();
			}
			delete jd;
		}
		
		curJobCount = m_JobCounter.Decrement();
	}

	m_Semaphore->Trigger();
}

bool CaptureSinkWorkerBase::execute(SCaptureSinkJobData &jobData)
{
	return false;
}
