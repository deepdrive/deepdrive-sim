

#include "DeepDrivePluginPrivatePCH.h"
#include "Private/Simulation/Agent/Controllers/DeepDriveAgentSplineDrivingCtrl.h"
#include "Private/Simulation/Misc/DeepDriveSplineTrack.h"
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


void DeepDriveAgentSplineDrivingCtrl::update(float dT, float desiredSpeed, float distanceToObstacle, float offset)
{
	if(m_Track && m_Agent)
	{
		//desiredSpeed = limitSpeed(desiredSpeed, distanceToObstacle);

		const float curSpeed = m_Agent->GetVehicleMovementComponent()->GetForwardSpeed();
		const float curSpeedKmh = curSpeed * 0.036f;
		const float eSpeed = desiredSpeed - curSpeedKmh;
		const float yThrottle = m_ThrottlePIDCtrl.advance(dT, eSpeed);
		m_curThrottle = FMath::Clamp(m_curThrottle + yThrottle * dT, -1.0f, 1.0f);
		m_Agent->SetThrottle(m_curThrottle);


		m_curAgentLocation = m_Agent->GetActorLocation();
		m_Track->setBaseLocation(m_curAgentLocation);

		const float lookAheadDist = 1000.0f; //  FMath::Max(1000.0f, curSpeed * 1.5f);
		FVector projLocAhead = m_Track->getLocationAhead(lookAheadDist, 0.0f);

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



FVector DeepDriveAgentSplineDrivingCtrl::getLookAheadPosOnSpline(const FVector &curAgentLocation, float lookAheadDistance, float offset)
{
	const float curKey = getLookAheadInputKey(lookAheadDistance);
	FVector posAhead = m_Spline->GetLocationAtSplineInputKey(curKey, ESplineCoordinateSpace::World);

	if (offset != 0.0f)
	{
		posAhead += m_Spline->GetTangentAtSplineInputKey(curKey, ESplineCoordinateSpace::World) * offset;
	}

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
	const float curSpeed = m_Agent->GetVehicleMovementComponent()->GetForwardSpeed();
	float curKey = getLookAheadInputKey(0.5 * curSpeed);	// m_Spline->FindInputKeyClosestToWorldLocation(m_curAgentLocation);
	float lastKey = getLookAheadInputKey(2.0 * curSpeed);

	float curveAngle = 0.0f;
	const int32 numSamples = 5;
	const float delta = (lastKey - curKey) / static_cast<float> (numSamples - 1);
	FVector2D lastDirection = FVector2D(m_Spline->GetDirectionAtSplineInputKey(curKey, ESplineCoordinateSpace::World));
	lastDirection.Normalize();
	for (int32 i = 0; i < numSamples; ++i)
	{
		curKey += delta;
		FVector2D curDirection = FVector2D(m_Spline->GetDirectionAtSplineInputKey(curKey, ESplineCoordinateSpace::World));
		curDirection.Normalize();

		curveAngle += FMath::RadiansToDegrees(FMath::Acos(FVector2D::DotProduct(curDirection, lastDirection)));
		lastDirection = curDirection;
	}

	const float fac = 0.8f * (1.0f - FMath::SmoothStep(30.0f, 90.0f, curveAngle)) + 0.2f;

	UE_LOG(LogDeepDriveAgentSplineDrivingCtrl, Log, TEXT("Curve angle %f Factor %f"), curveAngle, fac);

	return desiredSpeed * fac;
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

void DeepDriveAgentSplineDrivingCtrl::setSpline(USplineComponent *spline)
{
	m_Spline = spline;
//	if(spline)
//		m_Track = new DeepDriveSplineTrack(*spline);
}
