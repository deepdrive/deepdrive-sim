

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

	virtual const FDeepDriveDataOut& getDeepDriveData() const;

private:

	bool beginCapture();

	ADeepDriveSimulation 					&m_DeepDriveSim;

	FDeepDriveDataOut						m_DeepDriveData;
	
	float									m_CaptureInterval = 0.0f;
	float									m_TimeToNextCapture = 0.0f;
};



inline const FDeepDriveDataOut& DeepDriveSimulationCaptureProxy::getDeepDriveData() const
{
	return m_DeepDriveData;
}
