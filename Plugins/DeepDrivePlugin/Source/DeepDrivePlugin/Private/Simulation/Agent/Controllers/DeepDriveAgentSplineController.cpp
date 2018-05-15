

#include "DeepDrivePluginPrivatePCH.h"

#include "Public/Simulation/Agent/Controllers/DeepDriveAgentSplineController.h"
#include "Private/Simulation/Agent/Controllers/DeepDriveAgentSplineDrivingCtrl.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"
#include "Public/Simulation/Misc/DeepDriveSplineTrack.h"

#include "WheeledVehicleMovementComponent.h"

DEFINE_LOG_CATEGORY(LogDeepDriveAgentSplineController);

ADeepDriveAgentSplineController::ADeepDriveAgentSplineController()
{
	m_ControllerName = "Spline Controller";
	m_isGameDriving = true;
}


bool ADeepDriveAgentSplineController::Activate(ADeepDriveAgent &agent)
{
	if (Track)
	{
		m_SplineDrivingCtrl = new DeepDriveAgentSplineDrivingCtrl(PIDSteering, PIDThrottle, FVector());
		if(m_SplineDrivingCtrl)
		{
			m_SplineDrivingCtrl->setTrack(Track);
			m_SplineDrivingCtrl->setAgent(&agent);

			m_Spline = Track->GetSpline();
			resetAgentPosOnSpline(agent);

			UE_LOG(LogDeepDriveAgentSplineController, Log, TEXT("ADeepDriveAgentSplineController::Activate Successfully initialized") );
		}
	}
	else
		UE_LOG(LogDeepDriveAgentSplineController, Error, TEXT("ADeepDriveAgentSplineController::Activate Didn't find spline") );


	return Track != 0 && m_SplineDrivingCtrl != 0 && Super::Activate(agent);
}

bool ADeepDriveAgentSplineController::ResetAgent()
{
	if(Track && m_Agent)
	{
		UE_LOG(LogDeepDriveAgentSplineController, Log, TEXT("Reset Agent") );
		m_Agent->reset();
		resetAgentPosOnSpline(*m_Agent);
		return true;
	}
	return false;
}

void ADeepDriveAgentSplineController::Tick( float DeltaSeconds )
{
	if (m_Agent && m_SplineDrivingCtrl)
	{
		m_SplineDrivingCtrl->update(DeltaSeconds, DesiredSpeed, 0.0f);

		const float curSpeed = m_Agent->GetVehicleMovementComponent()->GetForwardSpeed();
		const float curSpeedKmh = curSpeed * 0.036f;
		const float eSpeed = DesiredSpeed - curSpeedKmh;

		addSpeedErrorSample(eSpeed);
		//UE_LOG(LogDeepDriveAgentSplineController, Log, TEXT("Distance Error: %f Speed Error: %f"), calcDistToCenterError(), calcSpeedError(eSpeed) );

	}
}

void ADeepDriveAgentSplineController::resetAgentPosOnSpline(ADeepDriveAgent &agent)
{
	FVector agentLocation = StartDistance > 0.0f ? (m_Spline->GetLocationAtDistanceAlongSpline(StartDistance, ESplineCoordinateSpace::World) + FVector(0.0f, 0.0f, 200.0f)) : agent.GetActorLocation();
	m_curDistanceOnSpline = getClosestDistanceOnSpline(agentLocation);
	FVector curPosOnSpline = m_Spline->GetLocationAtDistanceAlongSpline(m_curDistanceOnSpline, ESplineCoordinateSpace::World);
	curPosOnSpline.Z = agentLocation.Z + 50.0f;

	FQuat quat = m_Spline->GetQuaternionAtDistanceAlongSpline(m_curDistanceOnSpline, ESplineCoordinateSpace::World);

	FTransform transform(quat.Rotator(), curPosOnSpline, FVector(1.0f, 1.0f, 1.0f));

	agent.SetActorTransform(transform, false, 0, ETeleportType::TeleportPhysics);
}

float ADeepDriveAgentSplineController::getClosestDistanceOnSpline(const FVector &location)
{
	float distance = 0.0f;

	const float closestKey = m_Spline->FindInputKeyClosestToWorldLocation(location);

	const int32 index0 = floor(closestKey);
	const int32 index1 = floor(closestKey + 1.0f);

	const float dist0 = m_Spline->GetDistanceAlongSplineAtSplinePoint(index0);
	const float dist1 = m_Spline->GetDistanceAlongSplineAtSplinePoint(index1);


	return FMath::Lerp(dist0, dist1, closestKey - static_cast<float> (index0));
}


float ADeepDriveAgentSplineController::calcDistToCenterError()
{
	FVector agentLocation = m_Agent->GetActorLocation();

	const float curKey = m_Spline->FindInputKeyClosestToWorldLocation(agentLocation);

	FVector pos = m_Spline->GetLocationAtSplineInputKey(curKey, ESplineCoordinateSpace::World);
	FVector tng = m_Spline->GetTangentAtSplineInputKey(curKey, ESplineCoordinateSpace::World);

	FVector delta = agentLocation - pos;
	delta.Z = 0.0f;
	float dist = 0.0f;
	if (FVector::DotProduct(delta, delta) > 0.001f)
	{
		dist = delta.Size();
		delta.Normalize();

		if (FVector::DotProduct(delta, tng) >= 0.0f)
			m_SumDistToCenter -= dist;
		else
			m_SumDistToCenter += dist;

		++m_numDistSamples;
	}
	return m_numDistSamples > 0 ? m_SumDistToCenter / static_cast<float> (m_numDistSamples) : 0.0f;
}

void ADeepDriveAgentSplineController::addSpeedErrorSample(float curSpeedError)
{
	curSpeedError = FMath::Abs(curSpeedError);

	m_SpeedDeviationSum += curSpeedError * curSpeedError;
	++m_numSpeedDeviation;
}

void ADeepDriveAgentSplineController::OnCheckpointReached()
{
	UE_LOG(LogDeepDriveAgentSplineController, Log, TEXT("Speed Deviation: %f %d"), FMath::Sqrt( m_SpeedDeviationSum / static_cast<float> (m_numSpeedDeviation)), m_numSpeedDeviation );
	m_SpeedDeviationSum = 0.0f;
	m_numSpeedDeviation = 0;
}

void ADeepDriveAgentSplineController::OnDebugTrigger()
{
	if (m_isPaused)
	{
		UGameplayStatics::SetGamePaused(GetWorld(), false);
		m_isPaused = false;
	}
	else
	{
		FVector agentLocation = m_Agent->GetActorLocation();
		const float curKey = m_Spline->FindInputKeyClosestToWorldLocation(agentLocation);

		UE_LOG(LogDeepDriveAgentSplineController, Log, TEXT("Current key: %f"), curKey);

		UGameplayStatics::SetGamePaused(GetWorld(), true);
		m_isPaused = true;
	}
}
