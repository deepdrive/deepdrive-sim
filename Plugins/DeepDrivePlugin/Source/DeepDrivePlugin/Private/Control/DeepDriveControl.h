
#pragma once

#include "Engine.h"

#include "Private/Capture/CaptureBufferPool.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveControl, Log, All);

class ADeepDriveControlProxy;
class DeepDriveControlWorker;

class DeepDriveControl
{
public:

	static DeepDriveControl& GetInstance();

	static void Destroy();

	void RegisterProxy(ADeepDriveControlProxy &proxy, const FString &sharedMemName, uint32 sharedMemSize);

	void UnregisterProxy(ADeepDriveControlProxy &proxy);

	const DeepDriveMessageHeader* getMessage();

private:

	DeepDriveControl();

	ADeepDriveControlProxy			*m_Proxy = 0;

	DeepDriveControlWorker			*m_Worker = 0;

	static DeepDriveControl			*theInstance;
};
