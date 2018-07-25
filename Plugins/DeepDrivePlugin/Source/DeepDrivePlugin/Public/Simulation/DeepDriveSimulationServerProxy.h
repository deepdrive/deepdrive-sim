

#pragma once

#include "Simulation/Agent/DeepDriveAgentControllerBase.h"
#include "Public/Server/IDeepDriveServerProxy.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveSimulationServerProxy, Log, All);

class ADeepDriveSimulation;
class UWorld;

class DeepDriveSimulationServerProxy	:	public IDeepDriveServerProxy
{
public:

	DeepDriveSimulationServerProxy(ADeepDriveSimulation &deepDriveSim);

	bool initialize(const FString &ipAddress, int32 port, UWorld *world);

	void update( float DeltaSeconds );

	void shutdown();

	/**
	*		IDeepDriveServerProxy methods
	*/

	virtual void RegisterClient(int32 ClientId, bool IsMaster);

	virtual void UnregisterClient(int32 ClientId, bool IsMaster);

	virtual int32 RegisterCaptureCamera(float FieldOfView, int32 CaptureWidth, int32 CaptureHeight, FVector RelativePosition, FVector RelativeRotation, const FString &Label);

	virtual bool RequestAgentControl();

	virtual void ReleaseAgentControl();

	virtual void ResetAgent();

	virtual void SetAgentControlValues(float steering, float throttle, float brake, bool handbrake);

	virtual void ConfigureSimulation(const SimulationConfiguration &cfg, const SimulationGraphicsSettings &graphicsSettings, bool initialConfiguration);

	virtual void SetSunSimulation(const SunSimulationSettings &sunSimSettings);

private:

	ADeepDriveSimulation 			&m_DeepDriveSim;
	bool							m_isActive = false;
	
};

