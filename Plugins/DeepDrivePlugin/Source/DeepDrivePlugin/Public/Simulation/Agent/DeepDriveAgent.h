

#pragma once

#include "WheeledVehicle.h"

#include "Public/Simulation/DeepDriveSimulationDefines.h"

#include "DeepDriveAgent.generated.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveAgent, Log, All);

struct DeepDriveDataOut;
class UCaptureCameraComponent;
class ADeepDriveAgentControllerBase;
class ADeepDriveSimulation;

/**
 * 
 */
UCLASS(Abstract)
class DEEPDRIVEPLUGIN_API ADeepDriveAgent : public AWheeledVehicle
{
	GENERATED_BODY()
	
public:

	ADeepDriveAgent();

	void initialize(ADeepDriveSimulation &sim);

	virtual void BeginPlay() override;

	virtual void Tick( float DeltaSeconds ) override;

	void setResetTransform(const FTransform &transform);

	int32 RegisterCaptureCamera(float fieldOfView, int32 captureWidth, int32 captureHeight, FVector relativePosition, FVector relativeRotation, const FString &label);

	void UnregisterCaptureCamera(uint32 camId);

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
	void OnCaptureCameraAdded(int32 CameraId, UTextureRenderTarget2D *SceneTexture, const FString &label);

	UFUNCTION(BlueprintImplementableEvent, Category = "HUD")
	void OnCaptureCameraRemoved(int32 CameraId);

	UFUNCTION(BlueprintImplementableEvent, Category = "HUD")
	void SetCaptureEncoding(int32 CameraId, EDeepDriveInternalCaptureEncoding CaptureEncoding, UTextureRenderTarget2D *DepthTexture);

	UFUNCTION(BlueprintCallable, Category = "Simulation")
	void OnLapStart();

	UFUNCTION(BlueprintCallable, Category = "Simulation")
	void OnLapFinished();

	UFUNCTION(BlueprintImplementableEvent, Category = "Simulation")
	void OnSimulationReset();

	UFUNCTION(BlueprintCallable, Category = "Simulation")
	void SetCenterOfTrackSpline(USplineComponent *Spline);

	UFUNCTION(BlueprintCallable, BlueprintImplementableEvent, Category = "Simulation")
	void UpdateLights(float Zenith);

	UFUNCTION(BlueprintImplementableEvent, Category = "Car")
	void OnBrakeLight(bool BrakeLightOn);

	UFUNCTION(BlueprintImplementableEvent, Category = "Car")
	void OnReverseLight(bool ReverseLightOn);

	UFUNCTION(BlueprintImplementableEvent, Category = "Rendering")
	void SetRenderMode(bool Simple);

	UFUNCTION(BlueprintImplementableEvent, Category = "Agents")
	void OnAgentCreated();

	UFUNCTION(BlueprintCallable, Category = "Agents")
	void OnCheckpointReached();

	UFUNCTION(BlueprintCallable, Category = "Agents")
	void OnDebugTrigger();

	UFUNCTION(BlueprintCallable, Category = "Agents")
	int32 GetAgentId();

	UFUNCTION(BlueprintCallable, Category = "Agents")
	EDeepDriveAgentState GetAgentState();

	UFUNCTION(BlueprintCallable, Category = "Agents")
	bool IsEgoAgent();

	UFUNCTION(BlueprintCallable, Category = "Agents")
	void SetSpeedRange(float MinSpeed, float MaxSpeed);

	UFUNCTION(BlueprintCallable, Category = "Agents")
	void SetDirectionIndicatorState(EDeepDriveAgentDirectionIndicatorState DirectionIndicator);

	UFUNCTION(BlueprintCallable, Category = "Agents")
	EDeepDriveAgentDirectionIndicatorState GetDirectionIndicatorState();

	void setCollisionMode(bool simple);
	void setCollisionVisibility(bool visible);

	bool setViewMode(int32 cameraId, const FString &viewModeName);

	void setIsGameDriving(bool isGameDriving);
	void reset();

	void beginCapture(DeepDriveDataOut &deepDriveData);

	float getSpeed() const;
	float getSpeedKmh() const;

	float getFrontBumperDistance() const;
	float getBackBumperDistance() const;
	float getWheelBase() const;

	void setNextAgent(ADeepDriveAgent *agent, float distance);
	void setPrevAgent(ADeepDriveAgent *agent, float distance);

	ADeepDriveAgent* getNextAgent(float maxDistance, float *distance = 0);
	ADeepDriveAgent* getPrevAgent(float maxDistance, float *distance = 0);

	void setAgentController(ADeepDriveAgentControllerBase *ctrl);

	UFUNCTION(BlueprintCallable, Category = "Agents")
	ADeepDriveAgentControllerBase *getAgentController();

protected:

	UPROPERTY(EditDefaultsOnly, Category = Cameras)
	USpringArmComponent					*ChaseCameraStick = 0;
	
	UPROPERTY(EditDefaultsOnly, Category = Cameras)
	UCameraComponent					*ChaseCamera = 0;
	
	UPROPERTY(EditDefaultsOnly, Category = Cameras)
	UCameraComponent					*InteriorCamera = 0;

	UPROPERTY(EditDefaultsOnly, Category = Cameras)
	USpringArmComponent					*OrbitCameraArm = 0;
	
	UPROPERTY(EditDefaultsOnly, Category = Cameras)
	UCameraComponent					*OrbitCamera = 0;

	UPROPERTY(EditDefaultsOnly, Category = Collision)
	USceneComponent						*CollisionRoot = 0;

	UPROPERTY(EditDefaultsOnly, Category = Collision)
	UBoxComponent						*CollisionSimpleBox = 0;

	UPROPERTY(EditDefaultsOnly, Category = Collision)
	UBoxComponent						*CollisionFrontCenterBumper = 0;

	UPROPERTY(EditDefaultsOnly, Category = Collision)
	UBoxComponent						*CollisionFrontLeftBumper = 0;

	UPROPERTY(EditDefaultsOnly, Category = Collision)
	UBoxComponent						*CollisionFrontRightBumper = 0;

	UPROPERTY(EditDefaultsOnly, Category = Collision)
	UBoxComponent						*CollisionFrontLeftFender = 0;

	UPROPERTY(EditDefaultsOnly, Category = Collision)
	UBoxComponent						*CollisionFrontRightFender = 0;

	UPROPERTY(EditDefaultsOnly, Category = Collision)
	UBoxComponent						*CollisionLeftDoor = 0;

	UPROPERTY(EditDefaultsOnly, Category = Collision)
	UBoxComponent						*CollisionRightDoor = 0;

	UPROPERTY(EditDefaultsOnly, Category = Collision)
	UBoxComponent						*CollisionRearCenterBumper = 0;

	UPROPERTY(EditDefaultsOnly, Category = Collision)
	UBoxComponent						*CollisionRearLeftBumper = 0;

	UPROPERTY(EditDefaultsOnly, Category = Collision)
	UBoxComponent						*CollisionRearRightBumper = 0;

	UPROPERTY(EditDefaultsOnly, Category = Collision)
	UBoxComponent						*CollisionRearLeftFender = 0;

	UPROPERTY(EditDefaultsOnly, Category = Collision)
	UBoxComponent						*CollisionRearRightFender = 0;

	UPROPERTY(EditDefaultsOnly, Category = Car)
	float								FrontBumperDistance = 0.0f;

	UPROPERTY(EditDefaultsOnly, Category = Car)
	float 								BackBumperDistance = 0.0f;

	UPROPERTY(EditDefaultsOnly, Category = Car)
	float 								WheelBase = 0.0f;

private:
	typedef TMap<uint32, UCaptureCameraComponent*>	CaptureCameraMap;

	UFUNCTION()
	void OnBeginOverlap(UPrimitiveComponent *OverlappedComponent, AActor *OtherActor, UPrimitiveComponent *OtherComp, int32 OtherBodyIndex, bool bFromSweep, const FHitResult &SweepResult);

	ADeepDriveSimulation				*m_Simulation;

	int32								m_AgentId;
	ADeepDriveAgent						*m_NextAgent = 0;
	float								m_DistanceToNextAgent = -1.0f;
	ADeepDriveAgent						*m_PrevAgent = 0;
	float								m_DistanceToPrevAgent = -1.0f;

	bool								m_isEgoAgent = false;
	bool								m_hasFocus = false;

	ADeepDriveAgentControllerBase		*m_AgentController = 0;

	EDeepDriveAgentDirectionIndicatorState	m_DirectionIndicator = EDeepDriveAgentDirectionIndicatorState::UNKNOWN;

	bool								m_SimpleCollisionMode = false;
	bool								m_CollisionVisible = false;

	CaptureCameraMap					m_CaptureCameras;
	
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

	static int32						s_nextAgentId;
};

inline void ADeepDriveAgent::initialize(ADeepDriveSimulation &sim)
{
	m_Simulation = &sim;
}

inline void ADeepDriveAgent::setResetTransform(const FTransform &transform)
{
	m_ResetTransform = transform;
}

inline void ADeepDriveAgent::setIsGameDriving(bool isGameDriving)
{
	m_isGameDriving = isGameDriving;
}

inline int32 ADeepDriveAgent::GetAgentId()
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

inline float ADeepDriveAgent::getWheelBase() const
{
	return WheelBase;
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

inline void ADeepDriveAgent::setAgentController(ADeepDriveAgentControllerBase *ctrl)
{
	m_AgentController = ctrl;
}
