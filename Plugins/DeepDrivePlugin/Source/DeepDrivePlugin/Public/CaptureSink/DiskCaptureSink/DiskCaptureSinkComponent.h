// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CaptureSink/CaptureSinkComponentBase.h"
#include "DiskCaptureSinkComponent.generated.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDiskCaptureSinkComponent, Log, All);

class CaptureSinkWorkerBase;
struct SCaptureSinkJobData;

/**
 * 
 */
UCLASS(meta=(BlueprintSpawnableComponent), Category = "DeepDrivePlugin")
class DEEPDRIVEPLUGIN_API UDiskCaptureSinkComponent : public UCaptureSinkComponentBase
{
	GENERATED_BODY()
	
public:

	UDiskCaptureSinkComponent();

	virtual void begin(double timestamp, uint32 sequenceNumber, const FDeepDriveDataOut &deepDriveData);

	virtual void setCaptureBuffer(int32 cameraId, EDeepDriveCameraType cameraType, CaptureBuffer &captureBuffer);

	virtual void flush();	


	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Destination)
	FString		BasePathOnWindows;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Destination)
	FString		BasePathOnLinux;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Destination)
	TMap<EDeepDriveCameraType, FString>		CameraTypePaths;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Destination)
	FString		BaseFileName;

private:

	CaptureSinkWorkerBase			*m_Worker = 0;
	SCaptureSinkJobData				*m_curJobData = 0;

	FString							*m_BasePath = 0;

};
