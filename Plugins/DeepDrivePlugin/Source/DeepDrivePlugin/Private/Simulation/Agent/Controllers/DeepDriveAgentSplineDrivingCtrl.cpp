

#include "DeepDrivePluginPrivatePCH.h"
#include "Private/Simulation/Agent/Controllers/DeepDriveAgentSplineDrivingCtrl.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"

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


void DeepDriveAgentSplineDrivingCtrl::update(float dT, float desiredSpeed, float distanceToObstacle)
{
	if(m_Spline && m_Agent)
	{
		desiredSpeed = limitSpeed(desiredSpeed, distanceToObstacle);

		const float curSpeed = m_Agent->GetVehicleMovementComponent()->GetForwardSpeed();
		const float curSpeedKmh = curSpeed * 0.036f;
		const float eSpeed = desiredSpeed - curSpeedKmh;
		const float yThrottle = m_ThrottlePIDCtrl.advance(dT, eSpeed);
		m_curThrottle = FMath::Clamp(m_curThrottle + yThrottle * dT, -1.0f, 1.0f);
		m_Agent->SetThrottle(m_curThrottle);


		m_curAgentLocation = m_Agent->GetActorLocation();

		const float lookAheadDist = 500.0f;//FMath::Max(MinLookAheadDistance, curSpeed * LookAheadTime);
		FVector projLocAhead = getLookAheadPosOnSpline(m_curAgentLocation, lookAheadDist);

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

		const float ySteering = m_SteeringPIDCtrl.advance(dT, delta);
		m_Agent->SetSteering(ySteering);

		// UE_LOG(LogDeepDriveAgentSplineDrivingCtrl, Log, TEXT("DeepDriveAgentSplineDrivingCtrl::update curThrottle %f"), m_curThrottle );

	}
}



FVector DeepDriveAgentSplineDrivingCtrl::getLookAheadPosOnSpline(const FVector &curAgentLocation, float lookAheadDistance)
{
	return m_Spline->GetLocationAtSplineInputKey(getLookAheadInputKey(lookAheadDistance), ESplineCoordinateSpace::World);

	const float curKey = m_Spline->FindInputKeyClosestToWorldLocation(curAgentLocation);

	const int32 index0 = floor(curKey);
	const int32 index1 = ceil(curKey);

	const float dist0 = m_Spline->GetDistanceAlongSplineAtSplinePoint(index0);
	const float dist1 = m_Spline->GetDistanceAlongSplineAtSplinePoint(index1);

	const float dist = (m_Spline->GetLocationAtSplinePoint(index1, ESplineCoordinateSpace::World) - m_Spline->GetLocationAtSplinePoint(index0, ESplineCoordinateSpace::World)).Size();

	const float relDistance = lookAheadDistance / dist;

	const float carryOver = curKey + relDistance - static_cast<float> (index1);

	// UE_LOG(LogDeepDriveAgentSplineDrivingCtrl, Log, TEXT("curKey %f i0 %d i1 %d relDist %f"), curKey, index0, index1, relDistance);

	FVector posAhead;
	if(carryOver > 0.0f)
	{
		lookAheadDistance -= dist * (static_cast<float> (index1) - curKey);
		const float newDist = (m_Spline->GetLocationAtSplinePoint((index1 + 1) % m_Spline->GetNumberOfSplinePoints(), ESplineCoordinateSpace::World) - m_Spline->GetLocationAtSplinePoint(index1, ESplineCoordinateSpace::World)).Size();
		const float newRelDist = lookAheadDistance / newDist;

		// UE_LOG(LogDeepDriveAgentSplineDrivingCtrl, Log, TEXT("new lookAhead %f -> newRelDist %f"), lookAheadDistance, newRelDist);
		posAhead = m_Spline->GetLocationAtSplineInputKey(static_cast<float> (index1) + newRelDist, ESplineCoordinateSpace::World);
	}
	else
		posAhead = m_Spline->GetLocationAtSplineInputKey(curKey + relDistance, ESplineCoordinateSpace::World);


	return posAhead;
}


float DeepDriveAgentSplineDrivingCtrl::limitSpeed(float desiredSpeed, float distanceToObstacle)
{
	const float ds0 = calcSpeedLimitForCollision(desiredSpeed, distanceToObstacle);
	const float ds1 = calcSpeedLimitForCurvature(desiredSpeed);

	return FMath::Min(ds0, ds1);
}


float DeepDriveAgentSplineDrivingCtrl::calcSpeedLimitForCollision(float desiredSpeed, float distanceToObstacle)
{
	return desiredSpeed;
}

float DeepDriveAgentSplineDrivingCtrl::calcSpeedLimitForCurvature(float desiredSpeed)
{
	UE_LOG(LogDeepDriveAgentSplineDrivingCtrl, Log, TEXT("Direction %s"), *(m_Spline->FindDirectionClosestToWorldLocation(m_Agent->GetActorLocation(), ESplineCoordinateSpace::World).ToString()) );

	return desiredSpeed;
}


float DeepDriveAgentSplineDrivingCtrl::getLookAheadInputKey(float lookAheadDistance)
{
	float key = m_Spline->FindInputKeyClosestToWorldLocation(m_curAgentLocation);

	while(true)
	{
		const int32 index0 = floor(key);
		const int32 index1 = ceil(key);

		const float dist0 = m_Spline->GetDistanceAlongSplineAtSplinePoint(index0);
		const float dist1 = m_Spline->GetDistanceAlongSplineAtSplinePoint(index1);

		const float dist = (m_Spline->GetLocationAtSplinePoint(index1, ESplineCoordinateSpace::World) - m_Spline->GetLocationAtSplinePoint(index0, ESplineCoordinateSpace::World)).Size();

		const float relDistance = lookAheadDistance / dist;

		const float carryOver = key + relDistance - static_cast<float> (index1);

		if(carryOver > 0.0f)
		{
			lookAheadDistance -= dist * (static_cast<float> (index1) - key);
			const float newDist = (m_Spline->GetLocationAtSplinePoint((index1 + 1) % m_Spline->GetNumberOfSplinePoints(), ESplineCoordinateSpace::World) - m_Spline->GetLocationAtSplinePoint(index1, ESplineCoordinateSpace::World)).Size();
			const float newRelDist = lookAheadDistance / newDist;
			key = static_cast<float> (index1) + newRelDist;
			if(newRelDist < 1.0f)
				break;
		}
		else
		{
			key += relDistance;
			break;
		}
	}

	return key;
}

