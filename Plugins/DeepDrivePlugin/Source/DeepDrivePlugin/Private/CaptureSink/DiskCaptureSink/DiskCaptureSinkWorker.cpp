
#include "DeepDrivePluginPrivatePCH.h"

#include "Private/CaptureSink/DiskCaptureSink/DiskCaptureSinkWorker.h"
#include "Private/Capture/CaptureBuffer.h"


#include "ImageHandling/Image.h"
#include "ImageHandling/BmpSaveHandler.h"

#include "Runtime/Engine/Public/ImageUtils.h"

DEFINE_LOG_CATEGORY(LogDiskCaptureSinkWorker);


DiskCaptureSinkWorker::DiskCaptureSinkWorker()
	:	CaptureSinkWorkerBase("DiskCaptureSinkWorker")
{
	UE_LOG(LogDeepDriveCapture, Log, TEXT("DiskCaptureSinkWorker created"));
}

DiskCaptureSinkWorker::~DiskCaptureSinkWorker()
{
}

bool DiskCaptureSinkWorker::execute(SCaptureSinkJobData &jobData)
{
	SDiskCaptureSinkJobData &diskSinkJobData = static_cast<SDiskCaptureSinkJobData&> (jobData);

	const UEnum* CamTypeEnum = FindObject<UEnum>(ANY_PACKAGE, TEXT("EDeepDriveCameraType"));

	for(SCaptureSinkBufferData &captureBufferData : diskSinkJobData.captures)
	{
		const EDeepDriveCameraType camType = captureBufferData.camera_type;
		const int32 camId = captureBufferData.camera_id;
		CaptureBuffer *captureBuffer = captureBufferData.capture_buffer;

		FString filePath;
		FString camTypePath = diskSinkJobData.camera_type_paths.Contains(camType) ? diskSinkJobData.camera_type_paths[camType] : "";

		if(camTypePath != "")
			filePath = FPaths::Combine(diskSinkJobData.base_path, camTypePath, diskSinkJobData.base_file_name) + FString::FromInt(diskSinkJobData.sequence_number) + ".bmp";
		else
			filePath = FPaths::Combine(diskSinkJobData.base_path, diskSinkJobData.base_file_name) + FString::FromInt(diskSinkJobData.sequence_number) + ".bmp";

		UE_LOG(LogDeepDriveCapture, Log, TEXT("DiskCaptureSinkWorker::execute type %s with id %d to store at %s"), *(CamTypeEnum ? CamTypeEnum->GetEnumName(static_cast<uint8> (camType)) : TEXT("<Invalid Enum>")), camId, *(filePath));

		saveAsBmp(*captureBuffer, filePath);

	}


	return true;
}

void DiskCaptureSinkWorker::saveAsBmp(CaptureBuffer &captureBuffer, const FString &fileName)
{
	CaptureBuffer::DataType dataType = captureBuffer.getDataType();

	deepdrive::Image img;

	const uint32 width = captureBuffer.getWidth();
	const uint32 height = captureBuffer.getHeight();
	if(dataType == CaptureBuffer::Float16)
	{
		const FFloat16 *f16Src = captureBuffer.getBuffer<FFloat16>();
		img.storeAsRGB(f16Src, width, height);
	}
	else if(dataType == CaptureBuffer::UnsignedByte)
	{
		img.storeAsRGB(captureBuffer.getBuffer<uint8>(), width, height);
	}

	if(img.getSizeInBytes() > 0)
	{
		deepdrive::BmpSaveHandler bmpSave;
		bmpSave.save(fileName, img);
	}
}
