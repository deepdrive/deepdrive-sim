

#pragma once

#include "WheeledVehicle.h"

#include "Public/Simulation/DeepDriveSimulationDefines.h"

#include "DeepDriveAgent.generated.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveAgent, Log, All);

class UCaptureCameraComponent;

/**
 * 
 */
UCLASS()
class DEEPDRIVEPLUGIN_API ADeepDriveAgent : public AWheeledVehicle
{
	GENERATED_BODY()
	
public:

	ADeepDriveAgent();

	virtual void BeginPlay() override;

	virtual void Tick( float DeltaSeconds ) override;

	void setResetTransform(const FTransform &transform);

	int32 RegisterCaptureCamera(float fieldOfView, int32 captureWidth, int32 captureHeight, FVector relativePosition, FVector relativeRotation, const FString &label);

	void SetControlValues(float steering, float throttle, float brake, bool handbrake);

	void SetSteering(float steering);

	void SetThrottle(float throttle);

	void SetBrake(float brake);

	void SetHandbrake(bool on);

	void ActivateCamera(EDeepDriveAgentCameraType cameraType);

	void SetOrbitCameraRotation(float pitch, float yaw);

	UFUNCTION(BlueprintImplementableEvent, Category = "HUD")
	void SetInstrumentsVisibility(bool Visible);

	UFUNCTION(BlueprintImplementableEvent, Category = "HUD")
	void OnCaptureCameraAdded(UTextureRenderTarget2D *CaptureTexture, const FString &label);

	UFUNCTION(BlueprintCallable, Category = "Simulation")
	void OnLapStart();

	UFUNCTION(BlueprintCallable, Category = "Simulation")
	void OnLapFinished();

	UFUNCTION(BlueprintCallable, Category = "Simulation")
	void SetCenterOfTrackSpline(USplineComponent *Spline);

	UFUNCTION(BlueprintCallable, BlueprintImplementableEvent, Category = "Simulation")
	void UpdateLights(float Zenith);

	void setIsGameDriving(bool isGameDriving);
	void reset();

	FVector getAngularVelocity() const;
	FVector getAcceleration() const;
	FVector getAngularAcceleration() const;
	float getSpeed() const;

	FVector getDimensions() const;

	float getSteering() const;
	float getThrottle() const;
	float getBrake() const;
	bool getHandbrake() const;

	int32 getNumberOfLaps() const;
	float getDistanceAlongRoute() const;
	float getDistanceToCenterOfTrack() const;
	bool getIsGameDriving() const;

protected:

	UPROPERTY(EditDefaultsOnly, Category = Cameras)
	UCameraComponent					*ChaseCamera = 0;
	
	UPROPERTY(EditDefaultsOnly, Category = Cameras)
	UCameraComponent					*InteriorCamera = 0;

	UPROPERTY(EditDefaultsOnly, Category = Cameras)
	USpringArmComponent					*OrbitCameraArm = 0;
	
	UPROPERTY(EditDefaultsOnly, Category = Cameras)
	UCameraComponent					*OrbitCamera = 0;


private:

	TArray<UCaptureCameraComponent*>	m_CaptureCameras;
	
	float								m_curSteering = 0.0f;
	float								m_curThrottle = 0.0f;
	float								m_curBrake = 0.0f;
	bool								m_curHandbrake;

	FVector								m_prevVelocity;
	FVector								m_AngularVelocity;
	FVector								m_prevAngularVelocity;
	FVector								m_Acceleration;
	FVector								m_AngularAcceleration;
	FVector								m_Dimensions;

	FTransform 							m_ResetTransform;

	USplineComponent					*m_CenterOfTrackSpline = 0;
	bool								m_isGameDriving;
	int32								m_NumberOfLaps = 0;
	bool								m_LapStarted = false;
};


inline void ADeepDriveAgent::setResetTransform(const FTransform &transform)
{
	m_ResetTransform = transform;
}

inline void ADeepDriveAgent::setIsGameDriving(bool isGameDriving)
{
	m_isGameDriving = isGameDriving;
}

inline float ADeepDriveAgent::getSteering() const
{
	return m_curSteering;
}

inline float ADeepDriveAgent::getThrottle() const
{
	return m_curSteering;
}

inline float ADeepDriveAgent::getBrake() const
{
	return m_curBrake;
}

inline bool ADeepDriveAgent::getHandbrake() const
{
	return m_curHandbrake;
}

inline FVector ADeepDriveAgent::getAngularVelocity() const
{
	return m_AngularVelocity;
}

inline FVector ADeepDriveAgent::getAcceleration() const
{
	return m_Acceleration;
}

inline FVector ADeepDriveAgent::getAngularAcceleration() const
{
	return m_AngularAcceleration;
}

inline FVector ADeepDriveAgent::getDimensions() const
{
	return m_Dimensions;
}

inline int32 ADeepDriveAgent::getNumberOfLaps() const
{
	return m_NumberOfLaps;
}

inline bool ADeepDriveAgent::getIsGameDriving() const
{
	return m_isGameDriving;
}