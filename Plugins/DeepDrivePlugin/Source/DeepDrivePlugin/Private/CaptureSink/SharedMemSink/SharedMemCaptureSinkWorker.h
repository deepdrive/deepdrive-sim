
#pragma once

#include "Private/CaptureSink/CaptureSinkWorkerBase.h"
#include "Public/DeepDriveData.h"


DECLARE_LOG_CATEGORY_EXTERN(LogSharedMemCaptureSinkWorker, Log, All);

class SharedMemory;

class SharedMemCaptureSinkWorker : public CaptureSinkWorkerBase
{

public:

	struct SSharedMemCaptureSinkJobData : public SCaptureSinkJobData
	{
		SSharedMemCaptureSinkJobData(double timestamp, uint32 seqNr, const FDeepDriveDataOut &deepDriveData)
			: SCaptureSinkJobData(timestamp, seqNr)
			, deep_drive_data(deepDriveData)
		{
		}

		FDeepDriveDataOut		deep_drive_data;
	};

	SharedMemCaptureSinkWorker(const FString &sharedMemName, uint32 maxSharedMemSize);
	virtual ~SharedMemCaptureSinkWorker();

protected:

	virtual bool execute(SCaptureSinkJobData &jobData);

private:

	SharedMemory			*m_SharedMemory = 0;

	float					m_TotalSavingTime = 0.0f;
	float					m_SaveCount = 0.0f;
	double					m_lastLoggingTimestamp = 0.0f;

};
