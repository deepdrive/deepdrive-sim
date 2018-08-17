
#pragma once

#include "Runtime/Sockets/Public/IPAddress.h"

struct SimulationConfiguration;
struct SimulationGraphicsSettings;
struct SunSimulationSettings;

class IDeepDriveServerProxy
{

public:	

	virtual void RegisterClient(int32 ClientId, bool IsMaster) = 0;

	virtual void UnregisterClient(int32 ClientId, bool IsMaster) = 0;

	virtual int32 RegisterCaptureCamera(float FieldOfView, int32 CaptureWidth, int32 CaptureHeight, FVector RelativePosition, FVector RelativeRotation, const FString &Label) = 0;

	virtual bool RequestAgentControl() = 0;

	virtual void ReleaseAgentControl() = 0;

	virtual void ResetAgent() = 0;

	virtual void SetAgentControlValues(float steering, float throttle, float brake, bool handbrake) = 0;

	virtual bool SetViewMode(int32 cameraId, const FString &viewMode) = 0;

};
