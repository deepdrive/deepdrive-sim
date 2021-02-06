

#include "Simulation/DeepDriveSimulationCaptureProxy.h"
#include "Simulation/DeepDriveSimulation.h"
#include "Simulation/Agent/DeepDriveAgent.h"


#include "Capture/DeepDriveCapture.h"

DEFINE_LOG_CATEGORY(LogDeepDriveSimulationCaptureProxy);

DeepDriveSimulationCaptureProxy::DeepDriveSimulationCaptureProxy(ADeepDriveSimulation &deepDriveSim, float captureInterval)
	:	m_DeepDriveSim(deepDriveSim)
	,	m_CaptureInterval(captureInterval)
{
	DeepDriveCapture::GetInstance().RegisterProxy(*this);
	UE_LOG(LogDeepDriveSimulationCaptureProxy, Log, TEXT("Capture Proxy registered"));
}

void DeepDriveSimulationCaptureProxy::update( float DeltaSeconds )
{
	DeepDriveCapture &deepDriveCapture = DeepDriveCapture::GetInstance();

	deepDriveCapture.HandleCaptureResult();

	if(m_CaptureInterval >= 0.0f)
	{
		m_TimeToNextCapture -= DeltaSeconds;

		if(m_TimeToNextCapture <= 0.0f)
		{
			if(beginCapture())
				DeepDriveCapture::GetInstance().Capture();

			m_TimeToNextCapture = m_CaptureInterval;
		}
	}
}

void DeepDriveSimulationCaptureProxy::shutdown()
{
	DeepDriveCapture::GetInstance().UnregisterProxy(*this);
	UE_LOG(LogDeepDriveSimulationCaptureProxy, Log, TEXT("Capture Proxy unregistered"));
}

bool DeepDriveSimulationCaptureProxy::beginCapture()
{
	ADeepDriveAgent *agent = m_DeepDriveSim.getCurrentAgent();

	if(agent)
	{
		agent->beginCapture(m_DeepDriveData);
	}

	return agent != 0;
}

/**
*		IDeepDriveCaptureProxyInterface methods
*/

TArray<UCaptureSinkComponentBase*>& DeepDriveSimulationCaptureProxy::getSinks()
{
	return m_DeepDriveSim.getCaptureSinks();
}
