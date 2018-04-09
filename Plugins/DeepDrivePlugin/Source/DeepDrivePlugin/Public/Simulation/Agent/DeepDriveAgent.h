

#pragma once

#include "GameFramework/WheeledVehicle.h"

#include "Public/Simulation/DeepDriveSimulationDefines.h"

#include "DeepDriveAgent.generated.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveAgent, Log, All);

/**
 * 
 */
UCLASS()
class DEEPDRIVEPLUGIN_API ADeepDriveAgent : public AWheeledVehicle
{
	GENERATED_BODY()
	
public:

	ADeepDriveAgent();

	void RegisterCaptureCamera(float fieldOfView, int32 captureWidth, int32 captureHeight, FVector relativePosition, FVector relativeRotation, const FString &label);

	void SetControlValues(float steering, float throttle, float brake, bool handbrake);

	void SetSteering(float steering);

	void SetThrottle(float throttle);

	void ActivateCamera(EDeepDriveAgentCameraType cameraType);

	void SetOrbitCameraRotation(float pitch, float yaw);

protected:

	UPROPERTY(EditDefaultsOnly, Category = Cameras)
	UCameraComponent					*ChaseCamera = 0;
	
	UPROPERTY(EditDefaultsOnly, Category = Cameras)
	UCameraComponent					*InteriorCamera = 0;

	UPROPERTY(EditDefaultsOnly, Category = Cameras)
	USpringArmComponent					*OrbitCameraArm = 0;
	
	UPROPERTY(EditDefaultsOnly, Category = Cameras)
	UCameraComponent					*OrbitCamera = 0;


	float								m_curSteering = 0.0f;
	float								m_curThrottle = 0.0f;

};
