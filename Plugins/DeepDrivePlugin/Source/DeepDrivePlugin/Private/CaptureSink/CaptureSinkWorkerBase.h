
#pragma once

#include "Engine.h"
#include "Runtime/Core/Public/HAL/Runnable.h"

struct SCaptureSinkBufferData
{
	SCaptureSinkBufferData(EDeepDriveCameraType camType, int32 camId, CaptureBuffer &captureBuffer)
		:	camera_type(camType)
		,	camera_id(camId)
		,	capture_buffer(&captureBuffer)
	{
	}

	EDeepDriveCameraType		camera_type;
	int32						camera_id;
	CaptureBuffer				*capture_buffer;
};

struct SCaptureSinkJobData
{
	SCaptureSinkJobData(double ts, uint32 seqNr)
		:	timestamp(ts)
		,	sequence_number(seqNr)
	{
	}

	virtual ~SCaptureSinkJobData()
	{
	}

	double								timestamp;
	uint32								sequence_number;
	TArray<SCaptureSinkBufferData>		captures;
};

class CaptureSinkWorkerBase	:	public FRunnable
{
public:

	CaptureSinkWorkerBase(const FString &name);
	virtual ~CaptureSinkWorkerBase();

	virtual bool Init();
	virtual uint32 Run();
	virtual void Stop();

	void process(SCaptureSinkJobData &jobData);


protected:

	virtual bool execute(SCaptureSinkJobData &jobData);

private:

	FRunnableThread					*m_WorkerThread;
	FEvent							*m_Semaphore;
	bool							m_isStopped;

	TQueue<SCaptureSinkJobData*>	m_JobDataQueue;


};
