

#include "DeepDrivePluginPrivatePCH.h"
#include "Private/Simulation/Agent/Controllers/DeepDriveAgentSpeedController.h"
#include "Public/Simulation/Misc/DeepDriveSplineTrack.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoute.h"
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

void DeepDriveAgentSpeedController::initialize(ADeepDriveAgent &agent, ADeepDriveRoute &route, float safetyDistanceFactor)
{
	m_Agent = &agent;
	m_Route = &route;
	m_SafetyDistanceFactor = safetyDistanceFactor;
}

void DeepDriveAgentSpeedController::reset()
{
	m_ThrottlePIDCtrl.reset();
	m_BrakePIDCtrl.reset();
	m_curThrottle = 0.0f;
}

// void DeepDriveAgentSpeedController::update(float dT, float desiredSpeed)
// {
// 	if(m_Agent)
// 	{
// 		float curBrake = 1.0f;
// 		float curThrottle = 1.0f;

// 		if (desiredSpeed > 0.0f)
// 		{
// 			const float curSpeed = m_Agent->GetVehicleMovementComponent()->GetForwardSpeed();

// 			const float curSpeedKmh = curSpeed * 0.036f;
// 			const float deltaSpeed = desiredSpeed - curSpeedKmh;
// 			const float eSpeed = deltaSpeed / desiredSpeed;

// 			const float yThrottle = m_ThrottlePIDCtrl.advance(dT, eSpeed) * dT;
// 			m_curThrottle = FMath::Clamp(m_curThrottle + yThrottle, 0.0f, 1.0f);

// 			const float throttleDampFac = FMath::SmoothStep(-0.025, 0.025, eSpeed);
// 			curThrottle = m_curThrottle * throttleDampFac;

// 			curBrake = 0.0f;

// 			// UE_LOG(LogDeepDriveAgentSpeedController, Log, TEXT("DeepDriveAgentSpeedController::update desiredSpeed %4.2f curSpeed %4.2f eSpeed %f curThrottle %f | %f curBrake %f yThrottle %f"), desiredSpeed, curSpeedKmh, eSpeed, curThrottle, throttleDampFac, curBrake, yThrottle);
// 		}

// 		m_Agent->SetThrottle(curThrottle);
// 		m_Agent->SetBrake(curBrake);
// 	}
// }

void DeepDriveAgentSpeedController::update(float dT, float desiredSpeed, float desiredDistance, float curDistance)
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

			// avoid slowing down when car is moving but no throttle is applied
			// this can happen when switching controllers
			if(curSpeed > 0.0f && m_curThrottle == 0.0f)
				m_curThrottle = 1.0f;

			const float yThrottle = m_ThrottlePIDCtrl.advance(dT, eSpeed) * dT;
			m_curThrottle = FMath::Clamp(m_curThrottle + yThrottle, 0.0f, 1.0f);

			const float throttleDampFac = FMath::SmoothStep(-0.025, 0.025, eSpeed);
			curThrottle = m_curThrottle * throttleDampFac;

			if(desiredDistance > 0.0f)
			{
				float relDistance = curDistance / desiredDistance;
				const float distDampFac = FMath::SmoothStep(0.25f, 2.0f, relDistance);
				curThrottle *= distDampFac;
				// if (m_Agent->GetName() == "DeepDriveAgent_AliceGT_C_0")
				//	UE_LOG(LogDeepDriveAgentSpeedController, Log, TEXT("DeepDriveAgentSpeedController::update Agent %s relDist %f distDampFac %f curThr %f"), *(m_Agent->GetName()), relDistance, distDampFac, curThrottle);
			}

			curBrake = 0.0f;


			// UE_LOG(LogDeepDriveAgentSpeedController, Log, TEXT("DeepDriveAgentSpeedController::update desiredSpeed %4.2f curSpeed %4.2f eSpeed %f curThrottle %f | %f curBrake %f dThrottle %f"), desiredSpeed, curSpeedKmh, eSpeed, curThrottle, throttleDampFac, curBrake, yThrottle);
		}

		m_Agent->SetThrottle(curThrottle);
		m_Agent->SetBrake(curBrake);
	}
}

void DeepDriveAgentSpeedController::brake(float strength)
{
	m_Agent->SetThrottle(0.0f);
	m_Agent->SetBrake(strength);
}

float DeepDriveAgentSpeedController::limitSpeedByTrack(float desiredSpeed, float speedBoost)
{
	// const float trackSpeedLimit = m_Track->getSpeedLimit(0.0f);
	const float trackSpeedLimit = m_Track ? m_Track->getSpeedLimit(0.0f) : m_Route->getSpeedLimit(m_Agent->getFrontBumperDistance());
	return trackSpeedLimit > 0.0f ? FMath::Min(desiredSpeed, trackSpeedLimit * speedBoost) : desiredSpeed;
}

#if 0
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
#endif
