

#pragma once

#include "CoreMinimal.h"
#include "Simulation/Misc/PIDController.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveAgentSteeringController, Log, All);

class ADeepDriveAgent;
class ADeepDriveSplineTrack;
class ADeepDriveRoute;

/**
 * 
 */
//UCLASS(Blueprintable)
class DeepDriveAgentSteeringController
{
public:

	DeepDriveAgentSteeringController(const FVector &pidSteeringParams);

	~DeepDriveAgentSteeringController();

	void reset();

	void initialize(ADeepDriveAgent &agent, ADeepDriveSplineTrack &track);

	void initialize(ADeepDriveAgent &agent, ADeepDriveRoute &route);

	void setRoute(ADeepDriveRoute &route);

	void update(float dT, float desiredSpeed, float offset);

private:

	FVector getLookAheadPosOnSpline(const FVector &curAgentLocation, float lookAheadDistance, float offset);
	float getLookAheadInputKey(float lookAheadDistance);

	ADeepDriveAgent					*m_Agent = 0;
	ADeepDriveSplineTrack			*m_Track = 0;
	ADeepDriveRoute					*m_Route = 0;


	PIDController					m_SteeringPIDCtrl;


	float							m_curSteering = 0.0f;
	float							m_desiredSteering = 0.0f;

};

