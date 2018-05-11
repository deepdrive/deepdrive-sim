

#pragma once

#include "CoreMinimal.h"
#include "Private/Simulation/Misc/PIDController.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveAgentSplineDrivingCtrl, Log, All);

class ADeepDriveAgent;
class USplineComponent;
class ADeepDriveSplineTrack;

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

	void setTrack(ADeepDriveSplineTrack *track);

	void update(float dT, float desiredSpeed, float distanceToObstacle, float offset);


private:

	FVector getLookAheadPosOnSpline(const FVector &curAgentLocation, float lookAheadDistance, float offset);
	float getLookAheadInputKey(float lookAheadDistance);

	float limitSpeed(float desiredSpeed, float distanceToObstacle);

	float calcSpeedLimitForCollision(float desiredSpeed, float distanceToObstacle);
	float calcSpeedLimitForCurvature(float desiredSpeed);

	ADeepDriveAgent					*m_Agent = 0;
	USplineComponent				*m_Spline = 0;
	ADeepDriveSplineTrack			*m_Track = 0;


	PIDController					m_SteeringPIDCtrl;
	PIDController					m_ThrottlePIDCtrl;
	PIDController					m_BrakePIDCtrl;


	FVector							m_curAgentLocation;
	float							m_curThrottle = 0.0f;
	float							m_curSteering = 0.0f;
	float							m_desiredSteering = 0.0f;
};


inline void DeepDriveAgentSplineDrivingCtrl::setAgent(ADeepDriveAgent *agent)
{
	m_Agent = agent;
}

inline void DeepDriveAgentSplineDrivingCtrl::setTrack(ADeepDriveSplineTrack *track)
{
	m_Track = track;
}
