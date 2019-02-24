

#include "DeepDrivePluginPrivatePCH.h"
#include "Private/Simulation/Agent/Controllers/DeepDriveAgentSteeringController.h"
#include "Public/Simulation/Misc/DeepDriveSplineTrack.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoute.h"
#include "WheeledVehicleMovementComponent.h"
#include "Runtime/Engine/Classes/Components/SplineComponent.h"

DEFINE_LOG_CATEGORY(LogDeepDriveAgentSteeringController);

DeepDriveAgentSteeringController::DeepDriveAgentSteeringController(const FVector &pidSteeringParams)
	:	m_SteeringPIDCtrl(pidSteeringParams.X, pidSteeringParams.Y, pidSteeringParams.Z)
{
}

DeepDriveAgentSteeringController::~DeepDriveAgentSteeringController()
{
}

void DeepDriveAgentSteeringController::initialize(ADeepDriveAgent &agent, ADeepDriveSplineTrack &track)
{
	m_Agent = &agent;
	m_Track = &track;
}

void DeepDriveAgentSteeringController::initialize(ADeepDriveAgent &agent, ADeepDriveRoute &route)
{
	m_Agent = &agent;
	m_Route = &route;
}

void DeepDriveAgentSteeringController::setRoute(ADeepDriveRoute &route)
{
	m_Route = &route;
}

void DeepDriveAgentSteeringController::reset()
{
	m_SteeringPIDCtrl.reset();
	m_curSteering = 0.0f;
	m_desiredSteering = 0.0f;
}


void DeepDriveAgentSteeringController::update(float dT, float desiredSpeed, float offset)
{
	if	(	m_Agent
		&&	(m_Track || m_Route)
		)
	{
		FVector curAgentLocation = m_Agent->GetActorLocation();
		const float lookAheadDist = 500.0f; //  FMath::Max(1000.0f, curSpeed * 1.5f);
		FVector projLocAhead =		m_Track
								?	m_Track->getLocationAhead(lookAheadDist, offset)
								:	m_Route->getLocationAhead(lookAheadDist, offset)
								;

		FVector desiredForward = projLocAhead - curAgentLocation;
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

		m_desiredSteering = m_SteeringPIDCtrl.advance(dT, delta) * dT;
		//m_curSteering = FMath::FInterpTo(m_curSteering, m_desiredSteering, dT, 4.0f);
		m_curSteering = FMath::Clamp(m_desiredSteering, -1.0f, 1.0f);
		//ySteering = FMath::SmoothStep(0.0f, 80.0f, FMath::Abs(delta)) * FMath::Sign(delta);

		m_Agent->SetSteering(m_curSteering);
		m_Agent->setIsGameDriving(true);

		// UE_LOG(LogDeepDriveAgentSteeringController, Log, TEXT("DeepDriveAgentSteeringController::update curSteering %f"), m_curSteering);

		// UE_LOG(LogDeepDriveAgentSteeringController, Log, TEXT("DeepDriveAgentSteeringController::update curThrottle %f"), m_curThrottle );

	}
}
