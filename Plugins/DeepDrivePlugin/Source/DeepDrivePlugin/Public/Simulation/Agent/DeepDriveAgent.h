

#pragma once

#include "GameFramework/WheeledVehicle.h"

#include "Public/Simulation/DeepDriveSimulationDefines.h"

#include "DeepDriveAgent.generated.h"


/**
 * 
 */
UCLASS()
class DEEPDRIVEPLUGIN_API ADeepDriveAgent : public AWheeledVehicle
{
	GENERATED_BODY()
	
public:

	void RegisterCaptureCamera(float fieldOfView, int32 captureWidth, int32 captureHeight, FVector relativePosition, FVector relativeRotation, const FString &label);

	void SetControlValues(float steering, float throttle, float brake, bool handbrake);

	void SetSteering(float steering);

	void SetThrottle(float throttle);

	void ActivateCamera(EDeepDriveAgentCameraType cameraType);

	
};
