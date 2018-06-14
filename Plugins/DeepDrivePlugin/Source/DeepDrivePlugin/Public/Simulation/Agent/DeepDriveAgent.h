

#pragma once

#include "WheeledVehicle.h"

#include "Public/Simulation/DeepDriveSimulationDefines.h"

#include "DeepDriveAgent.generated.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveAgent, Log, All);

class UCaptureCameraComponent;
class ADeepDriveAgentControllerBase;

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

	void DeactivateCameras();

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

	UFUNCTION(BlueprintCallable, Category = "Agents")
	void OnCheckpointReached();

	UFUNCTION(BlueprintCallable, Category = "Agents")
	void OnDebugTrigger();

	float getDistanceToObstacleAhead(float maxDistance);

	void setIsGameDriving(bool isGameDriving);
	void reset();

	FVector getAngularVelocity() const;
	FVector getAcceleration() const;
	FVector getAngularAcceleration() const;
	float getSpeed() const;
	float getSpeedKmh() const;

	FVector getDimensions() const;

	float getSteering() const;
	float getThrottle() const;
	float getBrake() const;
	bool getHandbrake() const;

	int32 getNumberOfLaps() const;
	float getDistanceAlongRoute() const;
	float getDistanceToCenterOfTrack() const;
	bool getIsGameDriving() const;

	float getFrontBumperDistance() const;
	float getBackBumperDistance() const;

	int32 getAgentId() const;
	void setNextAgent(ADeepDriveAgent *agent, float distance);
	void setPrevAgent(ADeepDriveAgent *agent, float distance);

	ADeepDriveAgent* getNextAgent(float maxDistance, float *distance = 0);
	ADeepDriveAgent* getPrevAgent(float maxDistance, float *distance = 0);

	ADeepDriveAgentControllerBase *getAgentController();

protected:

	UPROPERTY(EditDefaultsOnly, Category = Cameras)
	UCameraComponent					*ChaseCamera = 0;
	
	UPROPERTY(EditDefaultsOnly, Category = Cameras)
	UCameraComponent					*InteriorCamera = 0;

	UPROPERTY(EditDefaultsOnly, Category = Cameras)
	USpringArmComponent					*OrbitCameraArm = 0;
	
	UPROPERTY(EditDefaultsOnly, Category = Cameras)
	UCameraComponent					*OrbitCamera = 0;

	UPROPERTY(EditDefaultsOnly, Category = Car)
	float								FrontBumperDistance = 0.0f;

	UPROPERTY(EditDefaultsOnly, Category = Car)
	float								BackBumperDistance = 0.0f;

private:

	int32								m_AgentId;
	ADeepDriveAgent						*m_NextAgent = 0;
	float								m_DistanceToNextAgent = 0.0f;
	ADeepDriveAgent						*m_PrevAgent = 0;
	float								m_DistanceToPrevAgent = 0.0f;

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

	static int32						s_nextAgentId;
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

inline int32 ADeepDriveAgent::getAgentId() const
{
	return m_AgentId;
}

inline float ADeepDriveAgent::getFrontBumperDistance() const
{
	return FrontBumperDistance;
}

inline float ADeepDriveAgent::getBackBumperDistance() const
{
	return BackBumperDistance;
}



inline void ADeepDriveAgent::setNextAgent(ADeepDriveAgent *agent, float distance)
{
	m_NextAgent = agent;
	m_DistanceToNextAgent = distance;
}

inline void ADeepDriveAgent::setPrevAgent(ADeepDriveAgent *agent, float distance)
{
	m_PrevAgent = agent;
	m_DistanceToPrevAgent = distance;
}

inline ADeepDriveAgent* ADeepDriveAgent::getNextAgent(float maxDistance, float *distance)
{
	if	(	maxDistance <= 0.0f
		||	m_DistanceToNextAgent <= maxDistance
		)
	{
		if (distance)
			*distance = m_DistanceToNextAgent;
		return m_NextAgent;
	}
	return 0;
}

inline ADeepDriveAgent* ADeepDriveAgent::getPrevAgent(float maxDistance, float *distance)
{
	if	(	maxDistance <= 0.0f
		||	m_DistanceToPrevAgent <= maxDistance
		)
	{
		if (distance)
			*distance = m_DistanceToPrevAgent;
		return m_PrevAgent;
	}
	return 0;
}

