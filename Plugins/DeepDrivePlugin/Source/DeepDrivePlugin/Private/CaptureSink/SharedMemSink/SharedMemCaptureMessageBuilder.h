
#pragma once

#include "Engine.h"

#include "Public/Capture/CaptureDefines.h"

DECLARE_LOG_CATEGORY_EXTERN(LogSharedMemCaptureMessageBuilder, Log, All);

class SharedMemory;
struct DeepDriveDataOut;
struct DeepDriveCaptureMessage;
struct DeepDriveCaptureCamera;
class CaptureBuffer;

class SharedMemCaptureMessageBuilder
{

public:

	SharedMemCaptureMessageBuilder(SharedMemory &sharedMem);

	void begin(const DeepDriveDataOut &deepDriveData, double timestamp, uint32 sequenceNumber);

	void addCamera(EDeepDriveCameraType camType, int32 camId, CaptureBuffer &sceneCaptureBuffer, CaptureBuffer *depthCaptureBuffer);

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
