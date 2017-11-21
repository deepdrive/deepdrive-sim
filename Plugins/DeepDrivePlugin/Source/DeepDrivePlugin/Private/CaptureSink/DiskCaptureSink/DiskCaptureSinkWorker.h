
#pragma once

#include "Private/CaptureSink/CaptureSinkWorkerBase.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDiskCaptureSinkWorker, Log, All);


class DiskCaptureSinkWorker	:	public CaptureSinkWorkerBase
{

public:

	struct SDiskCaptureSinkJobData : public SCaptureSinkJobData
	{
		SDiskCaptureSinkJobData(double timestamp, uint32 seqNr, const FString &basePath, const TMap<EDeepDriveCameraType, FString> &camTypePaths, const FString &baseFileName)
			: SCaptureSinkJobData(timestamp, seqNr)
			, base_path(basePath)
			, camera_type_paths(camTypePaths)
			, base_file_name(baseFileName)
		{
		}

		FString									base_path;
		TMap<EDeepDriveCameraType, FString>		camera_type_paths;
		FString									base_file_name;
	};

	DiskCaptureSinkWorker();
	virtual ~DiskCaptureSinkWorker();

protected:

	virtual bool execute(SCaptureSinkJobData &jobData);

private:

	void saveAsBmp(CaptureBuffer &captureBuffer, const FString &fileName);

};


