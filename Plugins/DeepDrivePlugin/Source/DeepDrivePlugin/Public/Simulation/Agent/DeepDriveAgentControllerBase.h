

#pragma once

#include "GameFramework/Controller.h"
#include "DeepDriveAgentControllerBase.generated.h"


DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveAgentControllerBase, Log, All);

class ADeepDriveSimulation;
class ADeepDriveAgent;
class USplineComponent;
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

	virtual void Deactivate();

	virtual void MoveForward(float axisValue);

	virtual void MoveRight(float axisValue);
	
	virtual void SetControlValues(float steering, float throttle, float brake, bool handbrake);
	
	virtual bool ResetAgent( /* const SimulationConfiguration &configuration */);
	
	virtual void OnAgentCollision(AActor *OtherActor, const FHitResult &HitResult, const FName &Tag);

	virtual void OnCheckpointReached();

	virtual void OnDebugTrigger();

	const FString& getControllerName() const;

	void getLastCollisionTime(FDateTime &utc, double &timeStamp, double &timeSinceLastCollision);

	bool updateAgentOnTrack();

protected:

	void activateController(ADeepDriveAgent &agent);

	bool initAgentOnTrack(ADeepDriveAgent &agent);
	
	void resetAgentPosOnSpline(ADeepDriveAgent &agent, USplineComponent *spline, float distance);
	float getClosestDistanceOnSpline(USplineComponent *spline, const FVector &location);

	ADeepDriveSimulation				*m_DeepDriveSimulation = 0;
	ADeepDriveAgent						*m_Agent = 0;

	ADeepDriveSplineTrack				*m_Track = 0;
	float								m_StartDistance = 0.0f;
	const float							m_LapDistanceThreshold = 500.0f;

	FString								m_ControllerName = "DeepDriveAgentControllerBase";
	bool								m_isGameDriving = false;

	bool								m_LapStarted = false;

	bool								m_hasCollisionOccured = false;
	FDateTime							m_lastCollisionTimeUTC = FDateTime();
	double								m_lastCollisionTimeStamp = 0.0;
	FName								m_CollisionLocation;
};


inline const FString& ADeepDriveAgentControllerBase::getControllerName() const
{
	return m_ControllerName;
}
