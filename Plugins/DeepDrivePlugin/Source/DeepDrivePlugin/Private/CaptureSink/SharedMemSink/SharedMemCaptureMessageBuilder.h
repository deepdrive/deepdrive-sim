
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

	void addCamera(EDeepDriveCameraType camType, int32 camId, CaptureBuffer &captureBuffer);

	void flush();

private:

	void decodeGrayscaleDepth(const FFloat16 *src, FFloat16 *colDst, FFloat16 *depthDst, uint32 width, uint32 height);

	void decodeYUVDepth(const FFloat16 *src, FFloat16 *colDst, FFloat16 *depthDst, uint32 width, uint32 height);

	void decodeRGBDepth(CaptureBuffer &captureBuffer, FFloat16 *colDst, FFloat16 *depthDst);

	void decodeSeparate(CaptureBuffer &sceneCaptureBuffer, CaptureBuffer &depthCaptureBuffer, FFloat16 *colDst, FFloat16 *depthDst);

	SharedMemory					&m_SharedMem;

	DeepDriveCaptureMessage			*m_Message = 0;

	uint32							m_MessageSize = 0;
	int32							m_remainingSize;

	DeepDriveCaptureCamera			*m_nextCamera = 0;
	DeepDriveCaptureCamera			*m_prevCamera = 0;
	uint32							m_prevCameraSize = 0;
};
