

#pragma once

#include "Simulation/Agent/DeepDriveAgentControllerBase.h"
#include "Public/Capture/IDeepDriveCaptureProxy.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveSimulationCaptureProxy, Log, All);

class ADeepDriveSimulation;

class DeepDriveSimulationCaptureProxy	:	public IDeepDriveCaptureProxy
{
public:

	DeepDriveSimulationCaptureProxy(ADeepDriveSimulation &deepDriveSim, float captureInterval);

	void update( float DeltaSeconds );

	void shutdown();

	/**
	*		IDeepDriveCaptureProxy methods
	*/

	virtual TArray<UCaptureSinkComponentBase*>& getSinks();

	virtual const DeepDriveDataOut& getDeepDriveData() const;

private:

	bool beginCapture();

	ADeepDriveSimulation 					&m_DeepDriveSim;

	DeepDriveDataOut						m_DeepDriveData;
	
	float									m_CaptureInterval = 0.0f;
	float									m_TimeToNextCapture = 0.0f;
};



inline const DeepDriveDataOut& DeepDriveSimulationCaptureProxy::getDeepDriveData() const
{
	return m_DeepDriveData;
}
