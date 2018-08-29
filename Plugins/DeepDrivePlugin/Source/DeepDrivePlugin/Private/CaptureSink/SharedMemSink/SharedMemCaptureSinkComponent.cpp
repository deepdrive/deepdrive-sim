// Fill out your copyright notice in the Description page of Project Settings.

#include "DeepDrivePluginPrivatePCH.h"
#include "DeepDrivePlugin.h"

#include "Public/CaptureSink/SharedMemSink/SharedMemCaptureSinkComponent.h"
#include "Private/CaptureSink/SharedMemSink/SharedMemCaptureSinkWorker.h"

DEFINE_LOG_CATEGORY(LogSharedMemCaptureSinkComponent);



USharedMemCaptureSinkComponent::USharedMemCaptureSinkComponent()
{
	m_Name = "SharedMemSink";
}

USharedMemCaptureSinkComponent::~USharedMemCaptureSinkComponent()
{
	delete m_Worker;
}


void USharedMemCaptureSinkComponent::BeginPlay()
{
	Super::BeginPlay();

	UE_LOG(LogSharedMemCaptureSinkComponent, Log, TEXT("USharedMemCaptureSinkComponent::InitializeComponent"));
	m_SharedMemoryName = UGameplayStatics::GetPlatformName() == "Linux" ? SharedMemNameLinux : SharedMemNameWindows;
	m_Worker = new SharedMemCaptureSinkWorker(m_SharedMemoryName, MaxSharedMemSize);
}

void USharedMemCaptureSinkComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	Super::EndPlay(EndPlayReason);

	UE_LOG(LogSharedMemCaptureSinkComponent, Log, TEXT("USharedMemCaptureSinkComponent::DestroyComponent"));
	m_Worker->kill();
}

void USharedMemCaptureSinkComponent::begin(double timestamp, uint32 sequenceNumber, const DeepDriveDataOut &deepDriveData)
{
	m_curJobData = new SharedMemCaptureSinkWorker::SSharedMemCaptureSinkJobData(timestamp, sequenceNumber, deepDriveData);
}

bool USharedMemCaptureSinkComponent::setCaptureBuffer(int32 cameraId, EDeepDriveCameraType cameraType, CaptureBuffer &sceneCaptureBuffer, CaptureBuffer *depthCaptureBuffer)
{
	bool res = false;
	if (m_curJobData)
	{
		m_curJobData->captures.Add(SCaptureSinkBufferData(cameraType, cameraId, sceneCaptureBuffer, depthCaptureBuffer));
		res = true;
	}
	return res;
}

void USharedMemCaptureSinkComponent::flush()
{
	if (m_curJobData
		&&	m_Worker
		)
	{
		m_Worker->process(*m_curJobData);
	}
}
