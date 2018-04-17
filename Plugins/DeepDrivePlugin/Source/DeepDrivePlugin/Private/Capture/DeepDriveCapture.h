
#pragma once

#include "Engine.h"

#include "Private/Capture/CaptureBufferPool.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveCapture, Log, All);

class UCaptureCameraComponent;
class IDeepDriveCaptureProxy;
struct SCaptureJob;
class USharedMemCaptureSinkComponent;

class DeepDriveCapture
{
	struct SCaptureComponentData
	{
		SCaptureComponentData(UCaptureCameraComponent *captureCmp)
			:	capture_component(captureCmp)
		{

		}

		UCaptureCameraComponent			*capture_component = 0;
	};

	typedef TMap<uint32, SCaptureComponentData>  CaptureComponentMap;

	struct SCycleTiming
	{
		SCycleTiming(double initialTS)
			: last_capture_timestamp(initialTS)
		{
		}

		double				last_capture_timestamp = 0.0;
		float				elapsed_capture_time = 0.0f;
		float				capture_count = 0.0f;
	};

public:

	static DeepDriveCapture& GetInstance();

	static void Destroy();

	void RegisterProxy(IDeepDriveCaptureProxy &proxy);

	void UnregisterProxy(IDeepDriveCaptureProxy &proxy);

	int32 RegisterCaptureComponent(UCaptureCameraComponent *captureComponent);
	
	void UnregisterCaptureComponent(int32 cameraId);

	void HandleCaptureResult();

	void Capture();

	USharedMemCaptureSinkComponent* getSharedMemorySink();

private:

	DeepDriveCapture();

	void reset();

	void processFinishedJobs();

	void processCapturing();

	static void executeCaptureJob(SCaptureJob &job);

	IDeepDriveCaptureProxy			*m_Proxy = 0;

	uint32							m_nextSequenceNumber = 1;

	int32							m_nextCaptureId = 1;
	CaptureComponentMap				m_CaptureComponentMap;

	TQueue<SCaptureJob*>			m_FinishedJobs;

	CaptureBufferPool				m_CaptureBufferPool;

	int32							m_curCycleIndex = 0;
	TMap<EDeepDriveCameraType, SCycleTiming>		m_CycleTimings;
	double							m_lastCaptureTS = 0.0;

	static float					m_TotalCaptureTime;
	static float					m_CaptureCount;
	static double					m_lastLoggingTimestamp;

	static DeepDriveCapture			*theInstance;
};
