

#include "DeepDrivePluginPrivatePCH.h"
#include "DeepDriveAgentSplineController.h"

#include "WheeledVehicleMovementComponent.h"

DEFINE_LOG_CATEGORY(LogDeepDriveAgentSplineController);

bool ADeepDriveAgentSplineController::Activate(ADeepDriveAgent &agent)
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

	return m_Spline != 0 && Super::Activate(agent);
}

void ADeepDriveAgentSplineController::Tick( float DeltaSeconds )
{
	FVector agentLocation = m_Agent->GetActorLocation();
	const float curDist = getClosestDistanceOnSpline(agentLocation);

	FVector projLocAhead = m_Spline->GetLocationAtDistanceAlongSpline(curDist + 1000.0f, ESplineCoordinateSpace::World);
	FVector2D desiredHeading(projLocAhead - agentLocation);
	FVector2D curHeading(m_Agent->GetActorForwardVector());
	FVector2D curRight(m_Agent->GetActorRightVector());

	desiredHeading.Normalize();

	float diff = 1.0f - FVector2D::DotProduct(desiredHeading, curHeading);

	if(FVector2D::DotProduct(curRight, desiredHeading) < 0.0f)
		diff *= -1.0f;

	UE_LOG(LogDeepDriveAgentControllerBase, Log, TEXT("diff %f"), diff );

	m_Agent->SetThrottle(0.5f);
	m_Agent->SetSteering(diff);



	/*
	if(m_Agent && UpdateSplineProgress())
		MoveAlongSpline();

	*/
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
