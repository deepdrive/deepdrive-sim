
#pragma once

#include "CoreMinimal.h"
#include "Simulation/Agent/Controllers/DeepDriveAgentTrafficAIController.h"
#include "DeepDriveAgentRemoteTrafficAIController.generated.h"


/**
 * 
 */
UCLASS()
class DEEPDRIVEPLUGIN_API ADeepDriveAgentRemoteTrafficAIController : public ADeepDriveAgentControllerBase
{
	GENERATED_BODY()

public:

	ADeepDriveAgentRemoteTrafficAIController();

	virtual void Tick(float DeltaSeconds) override;

	virtual void SetControlValues(float steering, float throttle, float brake, bool handbrake);

	virtual bool Activate(ADeepDriveAgent &agent, bool keepPosition);

	virtual bool ResetAgent();

	virtual float getDistanceAlongRoute();

	virtual float getRouteLength();

	virtual float getDistanceToCenterOfTrack();

	UFUNCTION(BlueprintCallable, Category = "Configuration")
	virtual void Configure(const FDeepDriveTrafficAIControllerConfiguration &Configuration, int32 StartPositionSlot, ADeepDriveSimulation* DeepDriveSimulation);

	UFUNCTION(BlueprintCallable, Category = "Configuration")
	virtual void ConfigureScenario(const FDeepDriveTrafficAIControllerConfiguration &Configuration, const FDeepDriveAgentScenarioConfiguration &ScenarioConfiguration, ADeepDriveSimulation* DeepDriveSimulation);

protected:

    UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "AIProps")
	FDeepDriveTrafficAIControllerConfiguration		m_Configuration;

private:

	enum State
	{
		Idle,
		ActiveRouteGuidance,
		Waiting
	};

	UPROPERTY()
	UBezierCurveComponent				*m_BezierCurve = 0;

	DeepDriveAgentSpeedController		*m_SpeedController = 0;

	SDeepDrivePathConfiguration			*m_PathConfiguration = 0;
	DeepDrivePathPlanner				*m_PathPlanner = 0;


	int32								m_StartIndex;

	float								m_DesiredSpeed = 0.0f;

	State								m_State = Idle;

	float								m_WaitTimer = 0.0f;

	bool								m_showPath = false;
};
