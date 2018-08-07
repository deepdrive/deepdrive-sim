
#pragma once

#include "Engine.h"
#include "Runtime/Core/Public/HAL/Runnable.h"
#include "Public/Capture/CaptureDefines.h"

class CaptureBuffer;

struct SCaptureSinkBufferData
{
	SCaptureSinkBufferData(EDeepDriveCameraType camType, int32 camId, CaptureBuffer &sceneCaptureBuffer)
		:	camera_type(camType)
		,	camera_id(camId)
		,	scene_capture_buffer(&sceneCaptureBuffer)
	{
	}

	EDeepDriveCameraType		camera_type;
	int32						camera_id;
	CaptureBuffer				*scene_capture_buffer;
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

	void kill();

protected:

	virtual bool execute(SCaptureSinkJobData &jobData);

private:

	FRunnableThread					*m_WorkerThread;
	FEvent							*m_Semaphore;
	bool							m_isStopped;

	TQueue<SCaptureSinkJobData*>	m_JobDataQueue;


};
