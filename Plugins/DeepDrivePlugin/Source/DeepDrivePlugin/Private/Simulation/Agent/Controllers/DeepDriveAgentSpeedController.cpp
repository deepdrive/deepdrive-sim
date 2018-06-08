

#include "DeepDrivePluginPrivatePCH.h"
#include "Private/Simulation/Agent/Controllers/DeepDriveAgentSpeedController.h"
#include "Public/Simulation/Misc/DeepDriveSplineTrack.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"
#include "WheeledVehicleMovementComponent.h"
#include "Runtime/Engine/Classes/Components/SplineComponent.h"

DEFINE_LOG_CATEGORY(LogDeepDriveAgentSpeedController);

DeepDriveAgentSpeedController::DeepDriveAgentSpeedController(const FVector &pidThrottleParams, const FVector &pidBrakeParams)
	:	m_ThrottlePIDCtrl(pidThrottleParams.X, pidThrottleParams.Y, pidThrottleParams.Z)
	,	m_BrakePIDCtrl(pidBrakeParams.X, pidBrakeParams.Y, pidBrakeParams.Z)
{
}

DeepDriveAgentSpeedController::~DeepDriveAgentSpeedController()
{
}

void DeepDriveAgentSpeedController::initialize(ADeepDriveAgent &agent, ADeepDriveSplineTrack &track, float safetyDistanceFactor)
{
	m_Agent = &agent;
	m_Track = &track;
	m_SafetyDistanceFactor = safetyDistanceFactor;
}


void DeepDriveAgentSpeedController::update(float dT, float desiredSpeed)
{
	if(m_Agent)
	{
		float curBrake = 1.0f;
		float curThrottle = 1.0f;

		if (desiredSpeed > 0.0f)
		{
			const float curSpeed = m_Agent->GetVehicleMovementComponent()->GetForwardSpeed();

			const float curSpeedKmh = curSpeed * 0.036f;
			const float deltaSpeed = desiredSpeed - curSpeedKmh;
			const float eSpeed = deltaSpeed / desiredSpeed;

			const float yThrottle = m_ThrottlePIDCtrl.advance(dT, eSpeed) * dT;
			m_curThrottle = FMath::Clamp(m_curThrottle + yThrottle, 0.0f, 1.0f);

			const float throttleDampFac = FMath::SmoothStep(-0.025, 0.025, eSpeed);
			curThrottle = m_curThrottle * throttleDampFac;

			curBrake = 0.0f;

			//UE_LOG(LogDeepDriveAgentSpeedController, Log, TEXT("DeepDriveAgentSpeedController::update desiredSpeed %4.2f curSpeed %4.2f eSpeed %f curThrottle %f | %f curBrake %f dThrottle %f"), desiredSpeed, curSpeedKmh, eSpeed, curThrottle, throttleDampFac, curBrake, yThrottle);
		}

		m_Agent->SetThrottle(curThrottle);
		m_Agent->SetBrake(curBrake);
	}
}

float DeepDriveAgentSpeedController::limitSpeedByTrack(float desiredSpeed, float speedBoost)
{
	const float trackSpeedLimit = m_Track->getSpeedLimit(0.0f);
	return trackSpeedLimit > 0.0f ? FMath::Min(desiredSpeed, trackSpeedLimit * speedBoost) : desiredSpeed;
}

float DeepDriveAgentSpeedController::limitSpeedByNextAgent(float desiredSpeed)
{
	float distanceToNext = 0.0f;
	ADeepDriveAgent *nextAgent = m_Agent->getNextAgent(&distanceToNext);
	if (nextAgent)
	{
		const float curSpeed = m_Agent->getSpeed();
		float safetyDistance = m_SafetyDistanceFactor * curSpeed * curSpeed / (2.0f * m_BrakingDeceleration);

		const float relDistance = distanceToNext / safetyDistance;
		if(relDistance < 1.0f)
		{
			const float nextAgentSpeed = nextAgent->getSpeedKmh();
			desiredSpeed = nextAgentSpeed * FMath::SmoothStep(0.0f, 1.0f, relDistance);
		}

		//if(nextAgentSpeed < desiredSpeed)
		//	desiredSpeed = FMath::Lerp(nextAgentSpeed, desiredSpeed, FMath::SmoothStep(1.0f, 1.25f, distanceToNext / safetyDistance));
	}

	return desiredSpeed;
}
