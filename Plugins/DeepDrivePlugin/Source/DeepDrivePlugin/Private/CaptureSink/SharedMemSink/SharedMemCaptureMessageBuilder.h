
#pragma once

#include "Engine.h"

DECLARE_LOG_CATEGORY_EXTERN(LogSharedMemCaptureMessageBuilder, Log, All);

class SharedMemory;
struct FDeepDriveDataOut;
struct DeepDriveCaptureMessage;
struct DeepDriveCaptureCamera;

class SharedMemCaptureMessageBuilder
{

public:

	SharedMemCaptureMessageBuilder(SharedMemory &sharedMem);

	void begin(const FDeepDriveDataOut &deepDriveData, double timestamp, uint32 sequenceNumber);

	void addCamera(EDeepDriveCameraType camType, int32 camId, CaptureBuffer &captureBuffer);

	void flush();

private:

	SharedMemory					&m_SharedMem;

	DeepDriveCaptureMessage			*m_Message = 0;

	uint32							m_MessageSize = 0;
	int32							m_remainingSize;

	DeepDriveCaptureCamera			*m_nextCamera = 0;
	DeepDriveCaptureCamera			*m_prevCamera = 0;
	uint32							m_prevCameraSize = 0;
};
