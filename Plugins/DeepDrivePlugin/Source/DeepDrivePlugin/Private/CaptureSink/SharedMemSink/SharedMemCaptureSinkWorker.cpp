
#include "DeepDrivePluginPrivatePCH.h"

#include "Private/CaptureSink/SharedMemSink/SharedMemCaptureSinkWorker.h"
#include "Private/CaptureSink/SharedMemSink/SharedMemCaptureMessageBuilder.h"

#include "Public/SharedMemory/SharedMemory.h"

DEFINE_LOG_CATEGORY(LogSharedMemCaptureSinkWorker);

SharedMemCaptureSinkWorker::SharedMemCaptureSinkWorker(const FString &sharedMemName, uint32 maxSharedMemSize)
	: CaptureSinkWorkerBase("SharedMemCaptureSinkWorker")
{
	UE_LOG(LogSharedMemCaptureSinkComponent, Log, TEXT("SharedMemCaptureSinkWorker::SharedMemCaptureSinkWorker"));
	m_SharedMemory = new SharedMemory();
	if (m_SharedMemory)
	{
		m_SharedMemory->create(sharedMemName, maxSharedMemSize);

		DeepDriveMessageHeader *message = reinterpret_cast<DeepDriveMessageHeader*> (m_SharedMemory->lockForWriting(-1));
		if(message)
		{
			message = new (message) DeepDriveMessageHeader(DeepDriveMessageType::Undefined, 0);
			message->setMessageId();
			m_SharedMemory->unlock(sizeof(DeepDriveMessageHeader));
		}
	}
}

SharedMemCaptureSinkWorker::~SharedMemCaptureSinkWorker()
{
	UE_LOG(LogSharedMemCaptureSinkComponent, Log, TEXT("SharedMemCaptureSinkWorker::~SharedMemCaptureSinkWorker"));
	delete m_SharedMemory;
}


bool SharedMemCaptureSinkWorker::execute(SCaptureSinkJobData &jobData)
{
	bool res = false;

	SSharedMemCaptureSinkJobData &sharedMemJobData = static_cast<SSharedMemCaptureSinkJobData&> (jobData);

	if(m_SharedMemory)
	{
		SharedMemCaptureMessageBuilder messageBuilder(*m_SharedMemory);

		const double before = FPlatformTime::Seconds();

		messageBuilder.begin(sharedMemJobData.deep_drive_data, sharedMemJobData.timestamp, sharedMemJobData.sequence_number);

		for(SCaptureSinkBufferData &captureBufferData : sharedMemJobData.captures)
		{
			const EDeepDriveCameraType camType = captureBufferData.camera_type;
			const int32 camId = captureBufferData.camera_id;
			CaptureBuffer *captureBuffer = captureBufferData.capture_buffer;

			if(captureBuffer)
			{
				messageBuilder.addCamera(camType, camId, *captureBuffer);
			}
		}

		messageBuilder.flush();

		const double after = FPlatformTime::Seconds();
		double duration = (after - before) * 1000.0;
		m_TotalSavingTime += static_cast<float> (duration);
		m_SaveCount = m_SaveCount + 1.0f;

		if (after - m_lastLoggingTimestamp > 10.0f && m_SaveCount > 1.0f)
		{
			UE_LOG(LogSharedMemCaptureSinkComponent, Log, TEXT("Saving in average took %f msecs"), m_TotalSavingTime / m_SaveCount);
			m_SaveCount = 0.0f;
			m_TotalSavingTime = 0.0f;
			m_lastLoggingTimestamp = after;
		}

	}

	return res;
}
