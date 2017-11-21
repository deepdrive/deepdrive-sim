// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "Components/ActorComponent.h"
#include "Public/Capture/CaptureCameraComponent.h"
#include "CaptureSinkComponentBase.generated.h"

class CaptureBuffer;
struct FDeepDriveDataOut;

UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent) )
class DEEPDRIVEPLUGIN_API UCaptureSinkComponentBase : public UActorComponent
{
	GENERATED_BODY()

public:	

	UCaptureSinkComponentBase();

	virtual void begin(double timestamp, uint32 sequenceNumber, const FDeepDriveDataOut &deepDriveData);

	virtual void setCaptureBuffer(int32 cameraId, EDeepDriveCameraType cameraType, CaptureBuffer &captureBuffer);

	virtual void flush();	

	const FString& getName() const;

protected:

	FString				m_Name = "CaptureSinkComponentBase";
	
};


inline const FString& UCaptureSinkComponentBase::getName() const
{
	return m_Name;
}
