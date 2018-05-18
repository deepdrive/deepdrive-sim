

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

	void initialize(ADeepDriveAgent &agent, ADeepDriveSplineTrack *track);

	void update(float dT, float desiredSpeed, float offset);


private:

	FVector getLookAheadPosOnSpline(const FVector &curAgentLocation, float lookAheadDistance, float offset);
	float getLookAheadInputKey(float lookAheadDistance);

	float limitSpeed(float desiredSpeed);

	float calcSpeedLimitForCollision(float desiredSpeed);
	float calcSpeedLimitForCurvature(float desiredSpeed);

	ADeepDriveAgent					*m_Agent = 0;
	ADeepDriveSplineTrack			*m_Track = 0;


	PIDController					m_SteeringPIDCtrl;
	PIDController					m_ThrottlePIDCtrl;
	PIDController					m_BrakePIDCtrl;


	FVector							m_curAgentLocation;
	float							m_curThrottle = 0.0f;
	float							m_curSteering = 0.0f;
	float							m_desiredSteering = 0.0f;
};

