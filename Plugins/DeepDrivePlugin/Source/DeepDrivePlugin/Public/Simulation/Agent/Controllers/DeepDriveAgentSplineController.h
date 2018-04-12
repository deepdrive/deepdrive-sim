

#pragma once

#include "Simulation/Agent/DeepDriveAgentControllerBase.h"
#include "Components/SplineComponent.h"
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
		}

		float advance(float dT, float curE, float kp, float ki, float kd)
		{
			m_sumE += curE;
			const float dE = (curE - m_prevE);

			float y = kp * curE + ki * dT * m_sumE;
			if(abs(dE) > 0.0000001)
				y += kd / (dT * curE);

			m_prevE = curE;

			return y;
		}

	private:

		float			m_Kp;
		float			m_Ki;
		float			m_Kd;

		float			m_prevE;
		float			m_sumE;

	};
	
public:

	ADeepDriveAgentSplineController();

	virtual void Tick( float DeltaSeconds ) override;

	virtual bool Activate(ADeepDriveAgent &agent);

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Control)
	float	MaxThrottle;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Control)
	FVector	PIDSteering;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Control)
	float	DesiredSpeed;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Control)
	FVector	PIDThrottle;


private:

	float getClosestDistanceOnSpline(const FVector &location);

	USplineComponent		*m_Spline = 0;

	PIDController			m_SteeringPIDCtrl;
	PIDController			m_ThrottlePIDCtrl;




	/**
		old implementation taken for ACar
	*/

	void MoveAlongSpline();

	bool UpdateSplineProgress();

	bool searchAlongSpline(FVector CurrentLocation, int step, float distToCurrent, float& distanceAlongRoute);

	bool getDistanceAlongSplineAtLocationWithStep(FVector CurrentLocation, unsigned int step, float& distanceAlongRoute);

	void GetDistanceAlongRouteAtLocation(FVector CurrentLocation);


	float					m_DistanceAlongRoute = 0.0f;
	float					m_DistanceToCenterOfLane = 0.0f;

	float					m_WaypointDistanceAlongSpline = 0.0f;
	float					m_WaypointStep = 400.0f;
	float					m_CloseDistanceThreshold = 1500.0f;

};
