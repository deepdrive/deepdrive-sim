

#include "DeepDrivePluginPrivatePCH.h"

#include "Public/Simulation/Agent/Controllers/DeepDriveAgentSplineController.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"

#include "WheeledVehicleMovementComponent.h"

DEFINE_LOG_CATEGORY(LogDeepDriveAgentSplineController);

ADeepDriveAgentSplineController::ADeepDriveAgentSplineController()
{
	m_ControllerName = "Spline Controller";
	m_isGameDriving = true;
}


bool ADeepDriveAgentSplineController::Activate(ADeepDriveAgent &agent)
{
	if(SplineActor)
	{
		TArray <UActorComponent*> splines = SplineActor->GetComponentsByClass( USplineComponent::StaticClass() );
		if(splines.Num() > 0)
			m_Spline = Cast<USplineComponent> (splines[0]);
	}

	if(m_Spline == 0)
	{
		/*
			A bit of a hack. Find spline to run on by searching for actors tagged with AgentSpline and having a spline component
		*/

		TArray<AActor*> actors;
		UGameplayStatics::GetAllActorsWithTag(GetWorld(), TEXT("AgentSpline"), actors);

		for(auto actor : actors)
		{
			TArray <UActorComponent*> splines = actor->GetComponentsByClass( USplineComponent::StaticClass() );
			if(splines.Num() > 0)
			{
				m_Spline = Cast<USplineComponent> (splines[0]);
				if(m_Spline)
					break;
			}
		}
	}

	if(m_Spline)
	{
		resetAgentPosOnSpline(agent);
	}
	else
		UE_LOG(LogDeepDriveAgentSplineController, Error, TEXT("ADeepDriveAgentSplineController::Activate Didn't find spline") );


	return m_Spline != 0 && Super::Activate(agent);
}

bool ADeepDriveAgentSplineController::ResetAgent()
{
	if(m_Spline && m_Agent)
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
	if (m_Agent && m_Spline)
	{
		if (1)
		{
			FVector agentLocation = m_Agent->GetActorLocation();

			updateDistanceOnSpline(agentLocation);

			const float curSpeed = m_Agent->GetVehicleMovementComponent()->GetForwardSpeed();
			const float lookAheadDist = FMath::Max(MinLookAheadDistance, curSpeed * LookAheadTime);
			FVector projLocAhead = m_Spline->GetLocationAtDistanceAlongSpline(m_curDistanceOnSpline + lookAheadDist, ESplineCoordinateSpace::World);
			projLocAhead = getLookAheadPosOnSpline(agentLocation, lookAheadDist);

			if (CurrentPosActor)
				CurrentPosActor->SetActorLocation(m_Spline->GetLocationAtDistanceAlongSpline(m_curDistanceOnSpline, ESplineCoordinateSpace::World) + FVector(0.0f, 0.0f, 200.0f));
			if (ProjectedPosActor)
				ProjectedPosActor->SetActorLocation(projLocAhead + FVector(0.0f, 0.0f, 200.0f));

			if(CurrentPosOnSpline)
			{
				const float curKey = m_Spline->FindInputKeyClosestToWorldLocation(agentLocation);
				const FVector posOnSpline = m_Spline->GetLocationAtSplineInputKey(curKey, ESplineCoordinateSpace::World);
				CurrentPosOnSpline->SetActorLocation(posOnSpline + FVector(0.0f, 0.0f, 200.0f));
			}

			FVector desiredForward = projLocAhead - agentLocation;
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


			m_projYawDelta = m_projYawDelta * 0.95f + 0.05f * delta;

			const float ySteering = m_SteeringPIDCtrl.advance(DeltaSeconds, delta, PIDSteering.X, PIDSteering.Y, PIDSteering.Z, 1.0f);
			m_Agent->SetSteering(ySteering);

			const float curSpeedKmh = curSpeed * 0.036f;
			const float eSpeed = DesiredSpeed - curSpeedKmh;
			const float yThrottle = m_ThrottlePIDCtrl.advance(DeltaSeconds, eSpeed, PIDThrottle.X, PIDThrottle.Y, PIDThrottle.Z, 0.75f);
			m_curThrottle = FMath::Clamp(m_curThrottle + yThrottle * DeltaSeconds * ThrottleFactor, -1.0f, 1.0f);
			m_Agent->SetThrottle(m_curThrottle);

			addSpeedErrorSample(eSpeed);
			//UE_LOG(LogDeepDriveAgentSplineController, Log, TEXT("Distance Error: %f Speed Error: %f"), calcDistToCenterError(), calcSpeedError(eSpeed) );

		}
		else if(m_Agent && UpdateSplineProgress())
			MoveAlongSpline();

	}
}



void ADeepDriveAgentSplineController::updateDistanceOnSpline(const FVector &curAgentLocation)
{
	FVector delta = curAgentLocation - m_prevAgentLocation;
	FVector tangent = m_Spline->GetTangentAtDistanceAlongSpline(m_curDistanceOnSpline, ESplineCoordinateSpace::World);
	FVector projected = delta.ProjectOnTo(tangent);

	m_curDistanceOnSpline += 1.005f * projected.Size();

	m_prevAgentLocation = curAgentLocation;
}

FVector ADeepDriveAgentSplineController::getLookAheadPosOnSpline(const FVector &curAgentLocation, float lookAheadDistance)
{
	const float curKey = m_Spline->FindInputKeyClosestToWorldLocation(curAgentLocation);

	const int32 index0 = floor(curKey);
	const int32 index1 = ceil(curKey);

	const float dist0 = m_Spline->GetDistanceAlongSplineAtSplinePoint(index0);
	const float dist1 = m_Spline->GetDistanceAlongSplineAtSplinePoint(index1);

	const float dist = (m_Spline->GetLocationAtSplinePoint(index1, ESplineCoordinateSpace::World) - m_Spline->GetLocationAtSplinePoint(index0, ESplineCoordinateSpace::World)).Size();

	const float relDistance = lookAheadDistance / dist;

	const float carryOver = curKey + relDistance - static_cast<float> (index1);

	// UE_LOG(LogDeepDriveAgentSplineController, Log, TEXT("curKey %f i0 %d i1 %d relDist %f"), curKey, index0, index1, relDistance);

	FVector posAhead;
	if(carryOver > 0.0f)
	{
		lookAheadDistance -= dist * (static_cast<float> (index1) - curKey);
		const float newDist = (m_Spline->GetLocationAtSplinePoint((index1 + 1) % m_Spline->GetNumberOfSplinePoints(), ESplineCoordinateSpace::World) - m_Spline->GetLocationAtSplinePoint(index1, ESplineCoordinateSpace::World)).Size();
		const float newRelDist = lookAheadDistance / newDist;

		// UE_LOG(LogDeepDriveAgentSplineController, Log, TEXT("new lookAhead %f -> newRelDist %f"), lookAheadDistance, newRelDist);
		posAhead = m_Spline->GetLocationAtSplineInputKey(static_cast<float> (index1) + newRelDist, ESplineCoordinateSpace::World);
	}
	else
		posAhead = m_Spline->GetLocationAtSplineInputKey(curKey + relDistance, ESplineCoordinateSpace::World);

	return posAhead;
}

void ADeepDriveAgentSplineController::resetAgentPosOnSpline(ADeepDriveAgent &agent)
{
	FVector agentLocation = agent.GetActorLocation();
	m_curDistanceOnSpline = getClosestDistanceOnSpline(agentLocation);
	FVector curPosOnSpline = m_Spline->GetLocationAtDistanceAlongSpline(m_curDistanceOnSpline, ESplineCoordinateSpace::World);
	curPosOnSpline.Z = agentLocation.Z + 50.0f;

	FQuat quat = m_Spline->GetQuaternionAtDistanceAlongSpline(m_curDistanceOnSpline, ESplineCoordinateSpace::World);

	FTransform transform(quat.Rotator(), curPosOnSpline, FVector(1.0f, 1.0f, 1.0f));

	agent.SetActorTransform(transform, false, 0, ETeleportType::TeleportPhysics);

	m_prevAgentLocation = agent.GetActorLocation();
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
	if (m_SpeedErrorSamples.Num() < m_maxSpeedErrorSamples)
	{
		m_SpeedErrorSamples.Add(curSpeedError);
	}
	else
	{
		m_SpeedErrorSamples[m_nextSpeedErrorSampleIndex] = curSpeedError;
		m_nextSpeedErrorSampleIndex = (m_nextSpeedErrorSampleIndex + 1) % m_maxSpeedErrorSamples;
	}

	m_totalSpeedError += curSpeedError;
	++m_numTotalSpeedErrorSamples;

	m_SpeedDeviationSum += curSpeedError * curSpeedError;
	++m_numSpeedDeviation;
}

void ADeepDriveAgentSplineController::OnCheckpointReached()
{
	float totalSpeedError = 0.0f;
	int32 numSamples = 0;
	for (auto &s : m_SpeedErrorSamples)
	{
		totalSpeedError += s;
		++numSamples;
	}

	float speedError = numSamples > 0 ? totalSpeedError / static_cast<float> (numSamples) : 0.0f;
	//UE_LOG(LogDeepDriveAgentSplineController, Log, TEXT("Speed Error: %f | %f"), speedError, m_totalSpeedError / static_cast<float> (m_numTotalSpeedErrorSamples) );

	UE_LOG(LogDeepDriveAgentSplineController, Log, TEXT("Speed Deviation: %f %d"), FMath::Sqrt( m_SpeedDeviationSum / static_cast<float> (m_numSpeedDeviation)), m_numSpeedDeviation );
	m_SpeedDeviationSum = 0.0f;
	m_numSpeedDeviation = 0;
}

bool ADeepDriveAgentSplineController::UpdateSplineProgress()
{
	FVector CurrentLocation = m_Agent->GetActorLocation();
	auto ClosestSplineLocation = m_Spline->FindLocationClosestToWorldLocation(CurrentLocation, ESplineCoordinateSpace::World);
	m_DistanceToCenterOfLane = sqrt(FVector::DistSquaredXY(CurrentLocation, ClosestSplineLocation));
	GetDistanceAlongRouteAtLocation(CurrentLocation);
	m_WaypointDistanceAlongSpline = static_cast<int>(m_DistanceAlongRoute + m_WaypointStep + m_CloseDistanceThreshold) / static_cast<int>(m_WaypointStep) * m_WaypointStep; // Assumes whole number waypoint step

	UE_LOG(LogTemp, VeryVerbose, TEXT("m_WaypointDistanceAlongSpline %f"), m_WaypointDistanceAlongSpline);

	FVector WaypointPosition = m_Spline->GetLocationAtDistanceAlongSpline(m_WaypointDistanceAlongSpline, ESplineCoordinateSpace::World);
	if (FVector::Dist(WaypointPosition, CurrentLocation) < m_CloseDistanceThreshold)
	{
		// We've gotten to the next waypoint along the spline
		m_WaypointDistanceAlongSpline += m_WaypointStep; // TODO: Don't assume we are travelling at speeds and framerates for this to make sense.
		auto SplineLength = m_Spline->GetSplineLength();
		if (m_WaypointDistanceAlongSpline > SplineLength)
		{
			UE_LOG(LogTemp, Warning, 
				TEXT("resetting target point on spline after making full trip around track (waypoint distance: %f, spline length: %f"), 
				m_WaypointDistanceAlongSpline, SplineLength);

			m_WaypointDistanceAlongSpline = 0.f;
			m_DistanceAlongRoute = 0.f;
			//LapNumber += 1;
		}
	}

	return true;
}

bool ADeepDriveAgentSplineController::searchAlongSpline(FVector CurrentLocation, int step, float distToCurrent, float& DistanceAlongRoute)
{	int i = 0;
	while (true)
	{
		FVector nextLocation = m_Spline->GetLocationAtDistanceAlongSpline(m_DistanceAlongRoute + step, ESplineCoordinateSpace::World);
		float distToNext = FVector::Dist(CurrentLocation, nextLocation);
		UE_LOG(LogTemp, VeryVerbose, TEXT("distToCurrent %f, distToNext %f, m_DistanceAlongRoute %f, step %d"), distToCurrent, distToNext, m_DistanceAlongRoute, step);

		if (distToNext > distToCurrent) // || m_DistanceAlongRoute <= static_cast<float> (abs(step))
		{
			return true;
		}
		else if (i > 9)
		{
			UE_LOG(LogTemp, VeryVerbose, TEXT("searched %d steps for closer spline point - giving up!"), i);
			return false;
		}
		else {
			UE_LOG(LogTemp, VeryVerbose, TEXT("advancing distance along route: %f by step %d"), m_DistanceAlongRoute, step);
			distToCurrent = distToNext;
			m_DistanceAlongRoute += step;
		}
		i++;
	}
}

bool ADeepDriveAgentSplineController::getDistanceAlongSplineAtLocationWithStep(FVector CurrentLocation, unsigned int step, float& DistanceAlongRoute)
{
	// Search the spline, starting from our previous distance, for locations closer to our current position. 
	// Then return the distance associated with that position.
	FVector prevLocation   = m_Spline->GetLocationAtDistanceAlongSpline(m_DistanceAlongRoute,        ESplineCoordinateSpace::World);
	FVector locationAhead  = m_Spline->GetLocationAtDistanceAlongSpline(m_DistanceAlongRoute + step, ESplineCoordinateSpace::World);
	FVector locationBehind = m_Spline->GetLocationAtDistanceAlongSpline(m_DistanceAlongRoute - step, ESplineCoordinateSpace::World);

	float distToPrev   = FVector::Dist(CurrentLocation, prevLocation);
	float distToAhead  = FVector::Dist(CurrentLocation, locationAhead);
	float distToBehind = FVector::Dist(CurrentLocation, locationBehind);
	UE_LOG(LogTemp, VeryVerbose, TEXT("distToPrev: %f, distToAhead: %f, distToBehind %f"), distToPrev, distToAhead, distToBehind);

	bool found = false;

	if (distToAhead <= distToPrev && distToAhead <= distToBehind)
	{
		// Move forward
		UE_LOG(LogTemp, VeryVerbose, TEXT("moving forward"));
		found = searchAlongSpline(CurrentLocation, step, distToAhead, m_DistanceAlongRoute);
	}
	else if (distToPrev <= distToAhead && distToPrev <= distToBehind)
	{
		// Stay
		UE_LOG(LogTemp, VeryVerbose, TEXT("staying"));
		found = true;
	}
	else if (distToBehind <= distToPrev && distToBehind <= distToAhead)
	{
		// Go back
		UE_LOG(LogTemp, VeryVerbose, TEXT("going back"));
		found = searchAlongSpline(CurrentLocation, - static_cast<int>(step), distToBehind, m_DistanceAlongRoute);
	}
	else
	{
		UE_LOG(LogTemp, Error, TEXT("Unexpected distance to waypoint! distToPrev: %f, distToAhead: %f, distToBehind %f"), distToPrev, distToAhead, distToBehind);
		// throw std::logic_error("Unexpected distance to waypoint!");
	}
	return found;
}

void ADeepDriveAgentSplineController::GetDistanceAlongRouteAtLocation(FVector CurrentLocation)
{
	// Search the spline's distance table starting at the last calculated distance: first at 1 meter increments, then 10cm

	//   TODO: Binary search
	if( ! getDistanceAlongSplineAtLocationWithStep(CurrentLocation, 100, m_DistanceAlongRoute))
	{
		// Our starting point way off base - search with larger steps
		getDistanceAlongSplineAtLocationWithStep(CurrentLocation, 1000, m_DistanceAlongRoute);

		getDistanceAlongSplineAtLocationWithStep(CurrentLocation, 100, m_DistanceAlongRoute);
	}
	UE_LOG(LogTemp, VeryVerbose, TEXT("dist 1 meter %f"), m_DistanceAlongRoute);

	// Narrow down search to get a more precise estimate
	getDistanceAlongSplineAtLocationWithStep(CurrentLocation, 10, m_DistanceAlongRoute);
	UE_LOG(LogTemp, VeryVerbose, TEXT("dist 100 cm %f"), m_DistanceAlongRoute);
}

void ADeepDriveAgentSplineController::MoveAlongSpline()
{
	// TODO: Poor man's motion planning - Look ahead n-steps and check tangential difference in yaw, pitch, and roll - then reduce velocity accordingly
	// TODO: 3D motion planning

	FVector CurrentLocation2 = m_Agent->GetActorLocation();
	FVector CurrentTarget = m_Spline->GetLocationAtDistanceAlongSpline(m_WaypointDistanceAlongSpline, ESplineCoordinateSpace::World);

	float CurrentYaw = m_Agent->GetActorRotation().Yaw;
	float DesiredYaw = FMath::Atan2(CurrentTarget.Y - CurrentLocation2.Y, CurrentTarget.X - CurrentLocation2.X) * 180 / PI;

	float YawDifference = DesiredYaw - CurrentYaw;

	if (YawDifference > 180)
	{
		YawDifference -= 360;
	}

	if (YawDifference < -180)
	{
		YawDifference += 360;
	}

	//const float ZeroSteeringToleranceDeg = 3.f;

	m_Agent->SetSteering(YawDifference / 30.f);

	auto speed = m_Agent->GetVehicleMovementComponent()->GetForwardSpeed();

	// throttle hysteresis
	float throttle = 0.0f;
	if(speed > 1800)
	{
		throttle = 0;
	}
	else if(speed < 1600)
	{
		throttle = 1.0f;
	}
	else
	{
		throttle = 0.75f;
	}

	/*if (FMath::Abs(YawDifference) < ZeroSteeringToleranceDeg)
	{
		SetSteering(0);
	}
	else
	{
		SetSteering(FMath::Sign(YawDifference));
	}*/

	m_Agent->SetThrottle(throttle);
}


