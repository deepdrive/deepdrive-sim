
#include "Capture/DeepDriveCapture.h"

#include "Capture/CaptureDefines.h"
#include "Capture/CaptureJob.h"
#include "Capture/CaptureBuffer.h"

#include "Capture/CaptureCameraComponent.h"
#include "Capture/IDeepDriveCaptureProxyInterface.h"
#include "CaptureSink/CaptureSinkComponentBase.h"
#include "CaptureSink/SharedMemSink/SharedMemCaptureSinkComponent.h"

DEFINE_LOG_CATEGORY(LogDeepDriveCapture);


DeepDriveCapture* DeepDriveCapture::theInstance = 0;
float DeepDriveCapture::m_TotalCaptureTime = 0.0f;
float DeepDriveCapture::m_CaptureCount = 0.0f;
double DeepDriveCapture::m_lastLoggingTimestamp = 0.0;


DeepDriveCapture& DeepDriveCapture::GetInstance()
{
	if(theInstance == 0)
	{
		theInstance = new DeepDriveCapture;
	}

	return *theInstance;
}

void DeepDriveCapture::Destroy()
{
	delete theInstance;
	theInstance = 0;
}


DeepDriveCapture::DeepDriveCapture()
{
}

void DeepDriveCapture::RegisterProxy(IDeepDriveCaptureProxyInterface &proxy)
{
	reset();

	m_lastCaptureTS = FPlatformTime::Seconds();
	m_Proxy = &proxy;
}

void DeepDriveCapture::UnregisterProxy(IDeepDriveCaptureProxyInterface &proxy)
{
	if(&proxy == m_Proxy)
		m_Proxy = 0;
}

int32 DeepDriveCapture::RegisterCaptureComponent(UCaptureCameraComponent *captureComponent)
{
	const int32 id = m_nextCaptureId++;

	m_CaptureComponentMap.Add(id, SCaptureComponentData(captureComponent));

	UE_LOG(LogDeepDriveCapture, Log, TEXT("Register CaptureCameraComponent with id %d"), id);

	m_CycleTimings.Add(captureComponent->CameraType, SCycleTiming(FPlatformTime::Seconds()));
	return id;
}

void DeepDriveCapture::UnregisterCaptureComponent(int32 cameraId)
{
	if (m_CaptureComponentMap.Contains(cameraId))
	{
		UE_LOG(LogDeepDriveCapture, Log, TEXT("Unregisterws CaptureCameraComponent with id %d"), cameraId);
		m_CaptureComponentMap.Remove(cameraId);
	}

}

void DeepDriveCapture::HandleCaptureResult()
{
	processFinishedJobs();
}

void DeepDriveCapture::Capture()
{
	processCapturing();
}

void DeepDriveCapture::reset()
{
	m_Proxy = 0;
	m_nextSequenceNumber = 1;

	m_nextCaptureId = 1;
	m_CaptureComponentMap.Empty();
	m_FinishedJobs.Empty();

	//m_CaptureBufferPool;

}


void DeepDriveCapture::processFinishedJobs()
{
	SCaptureJob *job = 0;
	if (	m_FinishedJobs.Dequeue(job)
		&&	job != 0
	   )
	{
		if(m_Proxy)
		{
			TArray<UCaptureSinkComponentBase*> &sinks =  m_Proxy->getSinks();

			for(UCaptureSinkComponentBase* &sink : sinks)
			{
				sink->begin(job->timestamp, job->sequence_number, m_Proxy->getDeepDriveData());
			}

			for(SCaptureRequest &captureReq : job->capture_requests)
			{
				CaptureBuffer *captureBuffer = captureReq.capture_buffer;

				if(captureBuffer)
				{
					for(UCaptureSinkComponentBase* &sink : sinks)
					{
						if(sink->setCaptureBuffer(captureReq.camera_id, captureReq.camera_type, *captureBuffer))
						{
							captureBuffer->addLock();
						}
					}
				}
			}

			for(UCaptureSinkComponentBase* &sink : sinks)
			{
				sink->flush();
			}

			for (SCaptureRequest &captureReq : job->capture_requests)
			{
				CaptureBuffer *captureBuffer = captureReq.capture_buffer;
				if (captureBuffer)
					captureBuffer->release();

			}

			m_onCaptureFinished.ExecuteIfBound(job->sequence_number);
			m_onCaptureFinished.Unbind();
		}

	}

	delete job;
}

void DeepDriveCapture::processCapturing()
{
	SCaptureJob *captureJob = new SCaptureJob;

/*
 *	Capture cycles currently deactivated
*/
#if 0
	const TArray< FCaptureCyle > &captureCycles = m_Proxy->CaptureCycles;
	if (captureCycles.Num())
	{
		const FCaptureCyle &cycle = captureCycles[m_curCycleIndex];
		m_curCycleIndex = (m_curCycleIndex + 1) % captureCycles.Num();
		for (auto &type : cycle.Cameras)
		{
			for (auto &captureCmp : m_CaptureComponentMap)
			{
				if (captureCmp.Value.capture_component->CameraType == type)
				{
					SCaptureRequest req;
					if (captureCmp.Value.capture_component->capture(req))
					{
						captureJob->capture_requests.Add(req);

						const UEnum* CamTypeEnum = FindObject<UEnum>(ANY_PACKAGE, TEXT("EDeepDriveCameraType"));

						SCycleTiming &timing = m_CycleTimings[type];
						const double now = FPlatformTime::Seconds();
						float delta = static_cast<float>(now - timing.last_capture_timestamp);
						timing.last_capture_timestamp = now;
						timing.elapsed_capture_time += delta * 1000.0f;
						timing.capture_count += 1.0f;

						UE_LOG(LogDeepDriveCapture, Log, TEXT("[%f] Capturing type %s with average frequency %f"), now,  *(CamTypeEnum ? CamTypeEnum->GetNameStringByIndex(static_cast<int32> (type)) : TEXT("<Invalid Enum>")), timing.elapsed_capture_time / timing.capture_count );
					}
					break;
				}
			}
		}
	}
	else
#endif
	{
		for (auto &captureCmp : m_CaptureComponentMap)
		{
			SCaptureRequest req;
			if (captureCmp.Value.capture_component->capture(req))
			{
				captureJob->capture_requests.Add(req);
			}
		}
	}

	if	(	captureJob->capture_requests.Num() > 0
		)
	{
		captureJob->timestamp = FPlatformTime::Seconds();
		captureJob->sequence_number = m_nextSequenceNumber++;
		captureJob->result_queue = &m_FinishedJobs;
		captureJob->capture_buffer_pool = &m_CaptureBufferPool;

		ENQUEUE_RENDER_COMMAND(ExecuteCaptureJob)
		(
			[captureJob](FRHICommandListImmediate &RHICmdList) {
				const double before = FPlatformTime::Seconds();

				DeepDriveCapture::executeCaptureJob(*captureJob);

				const double after = FPlatformTime::Seconds();
				double duration = after - before;
				DeepDriveCapture::m_TotalCaptureTime += static_cast<float>(duration * 1000.0);
				DeepDriveCapture::m_CaptureCount = DeepDriveCapture::m_CaptureCount + 1.0f;

				if (after - DeepDriveCapture::m_lastLoggingTimestamp > 10.0f && DeepDriveCapture::m_CaptureCount > 1.0f)
				{
					UE_LOG(LogDeepDriveCapture, Log, TEXT("Average capturing %f msecs"), DeepDriveCapture::m_TotalCaptureTime / DeepDriveCapture::m_CaptureCount);
					DeepDriveCapture::m_CaptureCount = 0.0f;
					DeepDriveCapture::m_TotalCaptureTime = 0.0f;
					DeepDriveCapture::m_lastLoggingTimestamp = after;
				}
			});

	}
	else
	{
		delete captureJob;
	}
}


void DeepDriveCapture::executeCaptureJob(SCaptureJob &job)
{
	for(SCaptureRequest &captureReq : job.capture_requests)
	{
		captureReq.capture_buffer = capture(*job.capture_buffer_pool, captureReq.scene_capture_source->TextureRHI->GetTexture2D());
		// UE_LOG(LogDeepDriveCapture, Log, TEXT("Capturing %d x %d  %p"), width, height, texture);

		captureReq.capture_buffer->setEncoding(captureReq.internal_capture_encoding);

		if(captureReq.depth_capture_source)
		{
			CaptureBuffer *depthCaptureBuffer = capture(*job.capture_buffer_pool, captureReq.depth_capture_source->TextureRHI->GetTexture2D());
			captureReq.capture_buffer->setSecondaryCaptureBuffer(depthCaptureBuffer);
		}
	}

	job.result_queue->Enqueue(&job);
}

CaptureBuffer* DeepDriveCapture::capture(CaptureBufferPool &pool, FRHITexture2D *srcTexture)
{
	EPixelFormat pixelFormat = srcTexture->GetFormat();
	uint32 width = srcTexture->GetSizeX();
	uint32 height = srcTexture->GetSizeY();

	uint32 stride;
	void *src = RHILockTexture2D(srcTexture, 0, RLM_ReadOnly, stride, false);

	CaptureBuffer *captureBuffer = pool.acquire(pixelFormat, width, height, stride);
	if(captureBuffer)
	{
		FMemory::BigBlockMemcpy(captureBuffer->getBuffer<void>(), src, captureBuffer->getBufferSize());
	}
	RHIUnlockTexture2D(srcTexture, 0, false);
	return captureBuffer;
}


USharedMemCaptureSinkComponent* DeepDriveCapture::getSharedMemorySink()
{
	USharedMemCaptureSinkComponent *sharedMemSink = 0;

	if (m_Proxy)
	{
		TArray<UCaptureSinkComponentBase*>&  sinks = m_Proxy->getSinks();
		for (auto &sink : sinks)
		{
			sharedMemSink = Cast<USharedMemCaptureSinkComponent>(sink);
			if (sharedMemSink)
				break;
		}
	}

	return sharedMemSink;
}

void DeepDriveCapture::onNextCapture(CaptureFinishedDelegate &captureFinished)
{
/*
	Ensure otherwise that a capture proxy is tickable when paused

	if(m_Proxy)
		m_Proxy->SetTickableWhenPaused(true);
*/
	m_onCaptureFinished = captureFinished;
}
