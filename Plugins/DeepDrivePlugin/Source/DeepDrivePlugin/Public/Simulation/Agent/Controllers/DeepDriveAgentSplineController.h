

#pragma once

#include "Simulation/Agent/DeepDriveAgentControllerBase.h"
#include "Components/SplineComponent.h"
#include "Runtime/Core/Public/GenericPlatform/GenericPlatformMath.h"
#include "DeepDriveAgentSplineController.generated.h"


DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveAgentSplineController, Log, All);

class DeepDriveAgentSplineDrivingCtrl;
class ADeepDriveSplineTrack;


/**
 * 
 */
UCLASS()
class DEEPDRIVEPLUGIN_API ADeepDriveAgentSplineController : public ADeepDriveAgentControllerBase
{
	GENERATED_BODY()

private:

public:

	ADeepDriveAgentSplineController();

	virtual void Tick( float DeltaSeconds ) override;

	virtual bool Activate(ADeepDriveAgent &agent);

	virtual bool ResetAgent();

	virtual void OnCheckpointReached();

	virtual void OnDebugTrigger();

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Spline)
	AActor	*SplineActor = 0;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Track)
	ADeepDriveSplineTrack	*Track = 0;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Control)
	float	LookAheadTime = 1.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Control)
	float	MinLookAheadDistance = 500.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Control)
	FVector	PIDSteering;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Control)
	float	DesiredSpeed;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Control)
	FVector	PIDThrottle;

private:

	void resetAgentPosOnSpline(ADeepDriveAgent &agent);

	float getClosestDistanceOnSpline(const FVector &location);

	float calcDistToCenterError();

	void addSpeedErrorSample(float curSpeedError);

	DeepDriveAgentSplineDrivingCtrl		*m_SplineDrivingCtrl;

	USplineComponent					*m_Spline = 0;
	float								m_curDistanceOnSpline = 0.0f;

	float								m_SumDistToCenter = 0.0f;
	int32								m_numDistSamples = 0;

	float								m_SumSpeedError = 0.0f;
	float								m_numSpeedSamples = 0;

	TArray<float>						m_SpeedErrorSamples;
	const int32							m_maxSpeedErrorSamples = 60;
	int32								m_nextSpeedErrorSampleIndex = 0;
	float								m_totalSpeedError = 0.0f;
	int32								m_numTotalSpeedErrorSamples = 0;

	float								m_SpeedDeviationSum = 0.0f;
	int32								m_numSpeedDeviation = 0;

	bool								m_isPaused = false;
};
