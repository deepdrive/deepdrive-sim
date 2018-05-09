

#pragma once

#include "CoreMinimal.h"
#include "Private/Simulation/Misc/PIDController.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveAgentSplineDrivingCtrl, Log, All);

class ADeepDriveAgent;
class USplineComponent;

/**
 * 
 */
class DeepDriveAgentSplineDrivingCtrl
{
public:

	DeepDriveAgentSplineDrivingCtrl(const FVector &pidSteeringParams, const FVector &pidThrottleParams, const FVector &pidBrakeParams);

	~DeepDriveAgentSplineDrivingCtrl();

	void setAgent(ADeepDriveAgent *agent);

	void setSpline(USplineComponent *spline);

	void update(float dT, float desiredSpeed, float distanceToObstacle);


private:

	FVector getLookAheadPosOnSpline(const FVector &curAgentLocation, float lookAheadDistance);
	float getLookAheadInputKey(float lookAheadDistance);

	float limitSpeed(float desiredSpeed, float distanceToObstacle);

	float calcSpeedLimitForCollision(float desiredSpeed, float distanceToObstacle);
	float calcSpeedLimitForCurvature(float desiredSpeed);

	ADeepDriveAgent					*m_Agent = 0;
	USplineComponent				*m_Spline = 0;

	PIDController					m_SteeringPIDCtrl;
	PIDController					m_ThrottlePIDCtrl;
	PIDController					m_BrakePIDCtrl;


	FVector							m_curAgentLocation;
	float							m_curThrottle = 0.0f;
};


inline void DeepDriveAgentSplineDrivingCtrl::setAgent(ADeepDriveAgent *agent)
{
	m_Agent = agent;
}


inline void DeepDriveAgentSplineDrivingCtrl::setSpline(USplineComponent *spline)
{
	m_Spline = spline;
}
