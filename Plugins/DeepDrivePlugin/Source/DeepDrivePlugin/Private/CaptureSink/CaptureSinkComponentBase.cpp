// Fill out your copyright notice in the Description page of Project Settings.

#include "CaptureSink/CaptureSinkComponentBase.h"


UCaptureSinkComponentBase::UCaptureSinkComponentBase()
{
}



void UCaptureSinkComponentBase::begin(double timestamp, uint32 sequenceNumber, const DeepDriveDataOut &deepDriveData)
{

}

bool UCaptureSinkComponentBase::setCaptureBuffer(int32 cameraId, EDeepDriveCameraType cameraType, CaptureBuffer &captureBuffer)
{
	return false;
}

void UCaptureSinkComponentBase::flush()
{

}
