
#include "DeepDrivePluginPrivatePCH.h"
#include "Private/Capture/DeepDriveCapture.h"

#include "Public/Capture/CaptureDefines.h"
#include "Private/Capture/CaptureJob.h"

#include "Public/Capture/CaptureCameraComponent.h"
#include "Public/Capture/DeepDriveCaptureProxy.h"
#include "Public/CaptureSink/CaptureSinkComponentBase.h"

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

void DeepDriveCapture::RegisterProxy(ADeepDriveCaptureProxy &proxy)
{
	reset();

	m_lastCaptureTS = FPlatformTime::Seconds();
	m_Proxy = &proxy;
}

void DeepDriveCapture::UnregisterProxy(ADeepDriveCaptureProxy &proxy)
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
						sink->setCaptureBuffer(captureReq.camera_id, captureReq.camera_type, *captureBuffer);
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
		}

	}

	delete job;
}

void DeepDriveCapture::processCapturing()
{
	SCaptureJob *captureJob = new SCaptureJob;

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

						UE_LOG(LogDeepDriveCapture, Log, TEXT("[%f] Capturing type %s with average frequency %f"), now,  *(CamTypeEnum ? CamTypeEnum->GetEnumName(static_cast<uint8> (type)) : TEXT("<Invalid Enum>")), timing.elapsed_capture_time / timing.capture_count );
					}
					break;
				}
			}
		}
	}
	else
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

		// TypeName, ParamType1, ParamName1, ParamValue1, Code
		ENQUEUE_UNIQUE_RENDER_COMMAND_ONEPARAMETER
		(
			ExecuteCaptureJob, SCaptureJob*, job, captureJob,
			{
				const double before = FPlatformTime::Seconds();
		
				DeepDriveCapture::executeCaptureJob(*job);

				const double after = FPlatformTime::Seconds();
				double duration = after - before;
				DeepDriveCapture::m_TotalCaptureTime += static_cast<float> (duration * 1000.0);
				DeepDriveCapture::m_CaptureCount = DeepDriveCapture::m_CaptureCount + 1.0f;

				if (after - DeepDriveCapture::m_lastLoggingTimestamp > 10.0f && DeepDriveCapture::m_CaptureCount > 1.0f)
				{
					UE_LOG(LogDeepDriveCapture, Log, TEXT("Average capturing %f msecs"), DeepDriveCapture::m_TotalCaptureTime / DeepDriveCapture::m_CaptureCount);
					DeepDriveCapture::m_CaptureCount = 0.0f;
					DeepDriveCapture::m_TotalCaptureTime = 0.0f;
					DeepDriveCapture::m_lastLoggingTimestamp = after;
				}
			}
		);

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
		FRHITexture2D *texture = captureReq.capture_source->TextureRHI->GetTexture2D();

		EPixelFormat pixelFormat = texture->GetFormat();
		uint32 width = texture->GetSizeX();
		uint32 height = texture->GetSizeY();

		uint32 stride;
		void *src = RHILockTexture2D(texture, 0, RLM_ReadOnly, stride, false);

		CaptureBuffer *captureBuffer = job.capture_buffer_pool->acquire(pixelFormat, width, height, stride);
		if(captureBuffer)
		{
			FMemory::BigBlockMemcpy(captureBuffer->getBuffer<void>(), src, captureBuffer->getBufferSize());
		}
		RHIUnlockTexture2D(texture, 0, false);

		// UE_LOG(LogDeepDriveCapture, Log, TEXT("Capturing %d x %d  %p"), width, height, texture);

		captureReq.capture_buffer = captureBuffer;
	}

	job.result_queue->Enqueue(&job);
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
