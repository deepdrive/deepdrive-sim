

#pragma once

#include "Simulation/Agent/DeepDriveAgentControllerBase.h"
#include "Components/SplineComponent.h"
#include "Runtime/Core/Public/GenericPlatform/GenericPlatformMath.h"
#include "DeepDriveAgentSplineController.generated.h"


DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveAgentSplineController, Log, All);

/**
 * 
 */
UCLASS()
class DEEPDRIVEPLUGIN_API ADeepDriveAgentSplineController : public ADeepDriveAgentControllerBase
{
	GENERATED_BODY()

private:

	class PIDController
	{
	public:

		PIDController()
			:	m_prevE(0.0f)
			,	m_sumE(0.0f)
		{
			for (signed i = 0; i < HistoryLength; ++i)
				m_History[i] = 0.0f;
		}

		float advance(float dT, float curE, float kp, float ki, float kd, float T)
		{
			const float dE = (curE - m_prevE);

			m_sumE -= m_History[m_lastHistoryIndex];
			m_sumE += curE;
			m_History[m_nextHistoryIndex] = curE;
			m_nextHistoryIndex = (m_nextHistoryIndex + 1) % HistoryLength;
			m_lastHistoryIndex = (m_lastHistoryIndex + 1) % HistoryLength;

			float y = kp * curE + ki * dT * m_sumE / static_cast<float> (HistoryLength) + dE * kd / dT;

			m_prevE = curE;

			return y;
		}

	private:

		float			m_Kp;
		float			m_Ki;
		float			m_Kd;

		float			m_prevE;
		float			m_sumE;

		enum
		{
			HistoryLength = 10
		};

		float			m_History[HistoryLength];
		int32			m_lastHistoryIndex = 0;
		int32			m_nextHistoryIndex = 1;
	};
	
public:

	ADeepDriveAgentSplineController();

	virtual void Tick( float DeltaSeconds ) override;

	virtual bool Activate(ADeepDriveAgent &agent);

	virtual bool ResetAgent();

	void OnCheckpointReached();


	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Spline)
	AActor	*SplineActor = 0;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Control)
	float	LookAheadTime = 1.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Control)
	float	MinLookAheadDistance = 500.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Control)
	FVector	PIDSteering;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Control)
	float	SteeringFactor = 1.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Control)
	float	DesiredSpeed;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Control)
	FVector	PIDThrottle;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Control)
	float	ThrottleFactor = 1.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Control)
	AActor*		CurrentPosActor;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Control)
	AActor*		ProjectedPosActor;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Control)
	AActor*		CurrentPosOnSpline;


private:

	void resetAgentPosOnSpline(ADeepDriveAgent &agent);

	float getClosestDistanceOnSpline(const FVector &location);

	void updateDistanceOnSpline(const FVector &curAgentLocation);

	FVector getLookAheadPosOnSpline(const FVector &curAgentLocation, float lookAheadDistance);

	USplineComponent		*m_Spline = 0;
	float					m_curDistanceOnSpline = 0.0f;

	FVector					m_prevAgentLocation;

	PIDController			m_SteeringPIDCtrl;
	PIDController			m_ThrottlePIDCtrl;

	float					m_curSteering = 0.0f;
	float					m_curThrottle = 0.0f;

	float					m_projYawDelta = 0.0f;



	/**
		old implementation taken for ACar
	*/

	void MoveAlongSpline();

	bool UpdateSplineProgress();

	bool searchAlongSpline(FVector CurrentLocation, int step, float distToCurrent, float& distanceAlongRoute);

	bool getDistanceAlongSplineAtLocationWithStep(FVector CurrentLocation, unsigned int step, float& distanceAlongRoute);

	void GetDistanceAlongRouteAtLocation(FVector CurrentLocation);

	float calcDistToCenterError();
	void addSpeedErrorSample(float curSpeedError);

	float					m_DistanceAlongRoute = 0.0f;
	float					m_DistanceToCenterOfLane = 0.0f;

	float					m_WaypointDistanceAlongSpline = 0.0f;
	float					m_WaypointStep = 400.0f;
	float					m_CloseDistanceThreshold = 1500.0f;


	float					m_SumDistToCenter = 0.0f;
	int32					m_numDistSamples = 0;

	float					m_SumSpeedError = 0.0f;
	float					m_numSpeedSamples = 0;

	TArray<float>			m_SpeedErrorSamples;
	const int32				m_maxSpeedErrorSamples = 60;
	int32					m_nextSpeedErrorSampleIndex = 0;
	float					m_totalSpeedError = 0.0f;
	int32					m_numTotalSpeedErrorSamples = 0;

	float					m_SpeedDeviationSum = 0.0f;
	int32					m_numSpeedDeviation = 0;
};
