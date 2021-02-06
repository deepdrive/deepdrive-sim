

#pragma once

#include "Simulation/Agent/DeepDriveAgentControllerBase.h"
#include "Server/IDeepDriveServerProxy.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveSimulationServerProxy, Log, All);

class ADeepDriveSimulation;
class UWorld;

class DeepDriveSimulationServerProxy	:	public IDeepDriveServerProxy
{
public:

	DeepDriveSimulationServerProxy(ADeepDriveSimulation &deepDriveSim);

	bool initialize(const FString &clientIPAddress, int32 clientPort, UWorld *world);

	void update( float DeltaSeconds );

	void shutdown();

	/**
	*		IDeepDriveServerProxy methods
	*/

	virtual void RegisterClient(int32 ClientId, bool IsMaster);

	virtual void UnregisterClient(int32 ClientId, bool IsMaster);

	virtual int32 RegisterCaptureCamera(float FieldOfView, int32 CaptureWidth, int32 CaptureHeight, FVector RelativePosition, FVector RelativeRotation, const FString &Label);

	virtual void UnregisterCaptureCamera(uint32 cameraId);

	virtual bool RequestAgentControl();

	virtual void ReleaseAgentControl();

	virtual void ResetAgent();

	virtual void SetAgentControlValues(float steering, float throttle, float brake, bool handbrake);

	virtual bool SetViewMode(int32 cameraId, const FString &viewMode);

private:

	ADeepDriveSimulation 			&m_DeepDriveSim;
	bool							m_isActive = false;
	
};

