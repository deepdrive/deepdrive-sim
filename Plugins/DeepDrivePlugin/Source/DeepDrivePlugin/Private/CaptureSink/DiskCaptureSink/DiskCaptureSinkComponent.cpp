// Fill out your copyright notice in the Description page of Project Settings.

#include "DeepDrivePluginPrivatePCH.h"

#include "DeepDrivePlugin.h"
#include "Public/CaptureSink/DiskCaptureSink/DiskCaptureSinkComponent.h"


#include "Public/Capture/CaptureDefines.h"
#include "Private/CaptureSink/DiskCaptureSink/DiskCaptureSinkWorker.h"

DEFINE_LOG_CATEGORY(LogDiskCaptureSinkComponent);


UDiskCaptureSinkComponent::UDiskCaptureSinkComponent()
{
	m_Name = "DiskCaptureSink";
	m_BasePath = UGameplayStatics::GetPlatformName() == "Linux" ? &BasePathOnLinux : &BasePathOnWindows;
}


void UDiskCaptureSinkComponent::begin(double timestamp, uint32 sequenceNumber, const FDeepDriveDataOut &deepDriveData)
{
	if(m_Worker == 0)
	{
		m_Worker = new DiskCaptureSinkWorker;
	}

	m_curJobData = new DiskCaptureSinkWorker::SDiskCaptureSinkJobData(timestamp, sequenceNumber, *m_BasePath, CameraTypePaths, BaseFileName);
	UE_LOG(LogDeepDriveCapture, Log, TEXT("UDiskCaptureSinkComponent::begin seqNr %d %p"), sequenceNumber, m_curJobData);
}

void UDiskCaptureSinkComponent::setCaptureBuffer(int32 cameraId, EDeepDriveCameraType cameraType, CaptureBuffer &captureBuffer)
{
	UE_LOG(LogDeepDriveCapture, Log, TEXT("UDiskCaptureSinkComponent::setCaptureBuffer %p Id %d"), m_curJobData, cameraId);

	if(m_curJobData)
	{
		m_curJobData->captures.Add( SCaptureSinkBufferData(cameraType, cameraId, captureBuffer) );
	}
}

void UDiskCaptureSinkComponent::flush()
{
	if	(	m_curJobData
		&&	m_Worker
		)
	{
		UE_LOG(LogDeepDriveCapture, Log, TEXT("UDiskCaptureSinkComponent::flush"));
		m_Worker->process(*m_curJobData);
	}
}
