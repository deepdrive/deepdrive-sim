

#pragma once

#include "GameFramework/Controller.h"

#include "DeepDriveData.h"
#include "Simulation/DeepDriveSimulationDefines.h"

#include "DeepDriveAgentControllerBase.generated.h"


DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveAgentControllerBase, Log, All);

class ADeepDriveSimulation;
class ADeepDriveAgent;
class USplineComponent;
class ADeepDriveRoute;
class ADeepDriveSplineTrack;

struct SimulationConfiguration;

/**
 * 
 */
UCLASS(Blueprintable, BlueprintType)
class DEEPDRIVEPLUGIN_API ADeepDriveAgentControllerBase : public AController
{
	GENERATED_BODY()
	
public:

	ADeepDriveAgentControllerBase();

	~ADeepDriveAgentControllerBase();

	virtual void OnConfigureSimulation(const SimulationConfiguration &configuration, bool initialConfiguration);

	virtual bool Activate(ADeepDriveAgent &agent, bool keepPosition);

	virtual void RequestControl();

	virtual void ReleaseControl();

	virtual void Reset();

	virtual void MoveForward(float axisValue);

	virtual void MoveRight(float axisValue);

	virtual void Brake(float axisValue);

	virtual void SetControlValues(float steering, float throttle, float brake, bool handbrake);

	virtual bool ResetAgent( /* const SimulationConfiguration &configuration */);

	virtual void OnRemoveAgent();

	virtual void SetSpeedRange(float MinSpeed, float MaxSpeed);

	virtual void OnCheckpointReached();

	virtual void OnDebugTrigger();

	virtual	float getDistanceAlongRoute();

	virtual	float getRouteLength();

	virtual	float getDistanceToCenterOfTrack();

	const FString& getControllerName() const;

	void OnAgentCollision(AActor *OtherActor, const FHitResult &HitResult, const FName &Tag);

	void getCollisionData(DeepDriveCollisionData &collisionDataOut);

	bool updateAgentOnTrack();

	void setIsPassing(bool isPassing);
	bool isPassing() const;

	void restoreStartPositionSlot(int32 startPositionSlot);

	FDeepDrivePath getPath();

	bool isRemotelyControlled() const;

protected:

	enum OperationMode
	{
		Standard,
		Scenario
	};

	struct ScenarionConfiguration
	{
		FVector			StartPosition;
		FVector			EndPosition;
	};

	void activateController(ADeepDriveAgent &agent);

	bool initAgentOnTrack(ADeepDriveAgent &agent);
	
	void resetAgentPosOnSpline(ADeepDriveAgent &agent, USplineComponent *spline, float distance);
	float getClosestDistanceOnSpline(USplineComponent *spline, const FVector &location);

	ADeepDriveSimulation				*m_DeepDriveSimulation = 0;
	ADeepDriveAgent						*m_Agent = 0;

	bool								m_isRemotelyControlled = false;

	OperationMode						m_OperationMode = OperationMode::Standard;
	ScenarionConfiguration				m_ScenarionConfiguration;

	UPROPERTY()
	ADeepDriveRoute						*m_Route = 0;

	ADeepDriveSplineTrack				*m_Track = 0;
	int32								m_StartPositionSlot = -1;
	float								m_StartDistance = 0.0f;
	const float							m_LapDistanceThreshold = 500.0f;  // cm? (everything is in cm in Unreal)

	FString								m_ControllerName = "DeepDriveAgentControllerBase";
	bool								m_isPassing = false;

	bool								m_LapStarted = false;

	bool								m_isCollisionEnabled = true;

	bool								m_hasCollisionOccured = false;
	DeepDriveCollisionData				m_CollisionData;

	float								m_InputTimer = -1.0f;

	const float							InputThreshold = 0.01f;
};


inline const FString& ADeepDriveAgentControllerBase::getControllerName() const
{
	return m_ControllerName;
}

inline void ADeepDriveAgentControllerBase::setIsPassing(bool isPassing)
{
	m_isPassing = isPassing;
}

inline bool ADeepDriveAgentControllerBase::isPassing() const
{
	return m_isPassing;
}

inline void ADeepDriveAgentControllerBase::restoreStartPositionSlot(int32 startPositionSlot)
{
	m_StartPositionSlot = startPositionSlot;
}

inline bool ADeepDriveAgentControllerBase::isRemotelyControlled() const
{
	return m_isRemotelyControlled;
}
