

#include "DeepDrivePluginPrivatePCH.h"
#include "Private/Simulation/Agent/Controllers/DeepDriveAgentSplineDrivingCtrl.h"
#include "Public/Simulation/Misc/DeepDriveSplineTrack.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"
#include "WheeledVehicleMovementComponent.h"
#include "Runtime/Engine/Classes/Components/SplineComponent.h"

DEFINE_LOG_CATEGORY(LogDeepDriveAgentSplineDrivingCtrl);

DeepDriveAgentSplineDrivingCtrl::DeepDriveAgentSplineDrivingCtrl(const FVector &pidSteeringParams, const FVector &pidThrottleParams, const FVector &pidBrakeParams)
	:	m_SteeringPIDCtrl(pidSteeringParams.X, pidSteeringParams.Y, pidSteeringParams.Z)
	,	m_ThrottlePIDCtrl(pidThrottleParams.X, pidThrottleParams.Y, pidThrottleParams.Z)
	,	m_BrakePIDCtrl(pidBrakeParams.X, pidBrakeParams.Y, pidBrakeParams.Z)
{
}

DeepDriveAgentSplineDrivingCtrl::~DeepDriveAgentSplineDrivingCtrl()
{
}

void DeepDriveAgentSplineDrivingCtrl::initialize(ADeepDriveAgent &agent, ADeepDriveSplineTrack *track)
{
	m_Agent = &agent;
	m_Track = track;

	m_Track->registerAgent(agent, m_Track->GetSpline()->FindInputKeyClosestToWorldLocation(agent.GetActorLocation()));
}


void DeepDriveAgentSplineDrivingCtrl::update(float dT, float desiredSpeed, float offset)
{
	if(m_Track && m_Agent)
	{
		desiredSpeed = limitSpeed(desiredSpeed);
		float curBrake = 1.0f;
		float curThrottle = 1.0f;

		if (desiredSpeed > 0.0f)
		{
			ADeepDriveAgent *nextAgent = 0;
			float distanceToNext = -1.0f;
			m_Track->getNextAgent(*m_Agent, nextAgent, distanceToNext);

			const float curSpeed = m_Agent->GetVehicleMovementComponent()->GetForwardSpeed();

			if (m_keepSafetyDistance)
			{
				const float BrakingDeceleration = 800.0f;
				float safetyDistance = 1.25f * curSpeed * curSpeed / (2.0f * BrakingDeceleration);

				safetyDistance *= m_SafetyDistanceFactor;

				if (distanceToNext < safetyDistance)
				{
					desiredSpeed = nextAgent->GetVehicleMovementComponent()->GetForwardSpeed() * 0.036f;
					distanceToNext = m_Agent->getDistanceToAgent(*nextAgent);
				}
			}

			const float curSpeedKmh = curSpeed * 0.036f;

			const float deltaSpeed = desiredSpeed - curSpeedKmh;
			const float eSpeed = deltaSpeed / desiredSpeed;

			const float yThrottle = m_ThrottlePIDCtrl.advance(dT, eSpeed) * dT;
			m_curThrottle = FMath::Clamp(m_curThrottle + yThrottle, 0.0f, 1.0f);

			if(eSpeed >= 0.0f)
				curThrottle = m_curThrottle;
			else
			{
				curThrottle = m_curThrottle * FMath::SmoothStep(-0.075, 0.0, eSpeed);
			}

			curBrake = 1.0f - FMath::SmoothStep(m_BrakingDistanceRange.X, m_BrakingDistanceRange.Y, distanceToNext);

			UE_LOG(LogDeepDriveAgentSplineDrivingCtrl, Log, TEXT("DeepDriveAgentSplineDrivingCtrl::update curSpeed %4.2f eSpeed %f curThrottle %f curBrake %f dThrottle %f"), curSpeed * 0.036f, eSpeed, curThrottle, curBrake, yThrottle);
		}

		m_Agent->SetThrottle(curThrottle);
		m_Agent->SetBrake(curBrake);

#if 0
		if (m_keepSafetyDistance)
		{
			ADeepDriveAgent *nextAgent = 0;
			float distance = 0.0f;
			m_Track->getNextAgent(*m_Agent, nextAgent, distance);

			const float BrakingDeceleration = 800.0f;
			float safetyDistance = 1.25f * curSpeed * curSpeed / (2.0f * BrakingDeceleration);

			safetyDistance *= m_SafetyDistanceFactor;

			if (distance < safetyDistance)
			{
				desiredSpeed = nextAgent->GetVehicleMovementComponent()->GetForwardSpeed() * 0.036f;

				distance = m_Agent->getDistanceToAgent(*nextAgent);
				desiredSpeed *= FMath::SmoothStep(m_BrakingDistanceRange.X, m_BrakingDistanceRange.Y, distance);

				brake = 1.0f - FMath::SmoothStep(m_BrakingDistanceRange.X, m_BrakingDistanceRange.Y, distance);

				UE_LOG(LogDeepDriveAgentSplineDrivingCtrl, Log, TEXT("DeepDriveAgentSplineDrivingCtrl::update distance %f safetyDist %f speed %f brake %f"), distance, safetyDistance, curSpeed, brake);
			}

		}
#endif

		m_curAgentLocation = m_Agent->GetActorLocation();
		m_Track->setBaseLocation(m_curAgentLocation);

		const float lookAheadDist = 1000.0f; //  FMath::Max(1000.0f, curSpeed * 1.5f);
		FVector projLocAhead = m_Track->getLocationAhead(lookAheadDist, offset);

		FVector desiredForward = projLocAhead - m_curAgentLocation;
		desiredForward.Normalize();

		float curYaw = m_Agent->GetActorRotation().Yaw;
		float desiredYaw = FMath::Atan2(desiredForward.Y, desiredForward.X) * 180.0f / PI;

		float delta = desiredYaw - curYaw;
		if (delta > 180.0f)
		{
			delta -= 360.0f;
		}

		if (delta < -180.0f)
		{
			delta += 360.0f;
		}

		m_desiredSteering = m_SteeringPIDCtrl.advance(dT, delta);
		m_curSteering = FMath::FInterpTo(m_curSteering, m_desiredSteering, dT, 4.0f);
		//ySteering = FMath::SmoothStep(0.0f, 80.0f, FMath::Abs(delta)) * FMath::Sign(delta);
		m_Agent->SetSteering(m_curSteering);

		// UE_LOG(LogDeepDriveAgentSplineDrivingCtrl, Log, TEXT("DeepDriveAgentSplineDrivingCtrl::update curThrottle %f"), m_curThrottle );

	}
}

float DeepDriveAgentSplineDrivingCtrl::limitSpeed(float desiredSpeed)
{
	const float trackSpeedLimit = m_Track->getSpeedLimit(0.0f);
	const float ds0 = trackSpeedLimit > 0.0f ? FMath::Min(desiredSpeed, trackSpeedLimit) : desiredSpeed;
	const float ds1 = calcSpeedLimitForCollision(ds0);

	//UE_LOG(LogDeepDriveAgentSplineDrivingCtrl, Log, TEXT("Track speed limit %f"), trackSpeedLimit );

	return ds1;
}


float DeepDriveAgentSplineDrivingCtrl::calcSpeedLimitForCollision(float desiredSpeed)
{
	const float dist2Obastacle = m_Agent->getDistanceToObstacleAhead(desiredSpeed);
	if (dist2Obastacle >= 0.0f)
	{

	}
	return desiredSpeed;
}
