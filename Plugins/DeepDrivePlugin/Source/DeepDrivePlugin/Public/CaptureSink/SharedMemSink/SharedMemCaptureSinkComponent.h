// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CaptureSink/CaptureSinkComponentBase.h"
#include "SharedMemCaptureSinkComponent.generated.h"


DECLARE_LOG_CATEGORY_EXTERN(LogSharedMemCaptureSinkComponent, Log, All);

class CaptureSinkWorkerBase;
struct SCaptureSinkJobData;


/**
 * 
 */
UCLASS(meta=(BlueprintSpawnableComponent), Category = "DeepDrivePlugin")
class DEEPDRIVEPLUGIN_API USharedMemCaptureSinkComponent : public UCaptureSinkComponentBase
{
	GENERATED_BODY()
	
	
public:

	USharedMemCaptureSinkComponent();
	
	virtual void BeginPlay() override;

	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

	virtual void begin(double timestamp, uint32 sequenceNumber, const FDeepDriveDataOut &deepDriveData);

	virtual void setCaptureBuffer(int32 cameraId, EDeepDriveCameraType cameraType, CaptureBuffer &captureBuffer);

	virtual void flush();

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = SharedMem)
	FString		SharedMemNameLinux;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = SharedMem)
	FString		SharedMemNameWindows;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = SharedMem)
	int32 MaxSharedMemSize = 150 * 1024 * 1024;

	const FString& getSharedMemoryName();

private:

	CaptureSinkWorkerBase			*m_Worker = 0;
	SCaptureSinkJobData				*m_curJobData = 0;

	FString							m_SharedMemoryName;

};


inline const FString& USharedMemCaptureSinkComponent::getSharedMemoryName()
{
	return m_SharedMemoryName;
}
