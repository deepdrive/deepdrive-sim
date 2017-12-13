// 

#include "DeepDrive.h"
#include "Vehicles/WheeledVehicleMovementComponent.h"
#include "Vehicles/WheeledVehicleMovementComponent4W.h"
#include "Car.h"
#include <stdexcept>

ACar::ACar()
{
	bIsGameDriving = true;
	bShouldReset = false;
	bPreviousShouldReset = false;
	bShouldResetPosition = false;
	bIsResetting = false;

	WaypointDistanceAlongSpline = 0.f;
	WaypointStep = 400.f;
	CloseDistanceThreshold = 1500.f;
	AIThrottle = 0.75f;
	LapNumber = 0;
	LastAppliedHandbrake = false;
}

void ACar::BeginPlay()
{
	Super::BeginPlay();
	EpisodeStartTime = GetAgentTime();
}

void ACar::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	if(bPreviousShouldReset)
	{
		// One tick has passed, all consumers of bShouldReset (i.e. AliceGT blueprint) have now had their chance to act on it
		bShouldReset = false;
	}

	if(bShouldReset)
	{
		UE_LOG(LogTemp, Warning, TEXT("Resetting Car"));

		WaypointDistanceAlongSpline = 0.f;
		DistanceAlongRoute = 0.f;
		DistanceToCenterOfLane = 0.f;
		LapNumber = 0;
		bShouldResetPosition = true;
		ResetAgentFinished();
	}
	bPreviousShouldReset = bShouldReset;

	FVector CurrentTickVelocity = GetVelocity();
	FVector CurrentAngularVelocity = GetMesh() != nullptr ? GetMesh()->GetPhysicsAngularVelocity() : FVector(0.f, 0.f, 0.f);

	if (DeltaTime > 0)
	{
		Acceleration = (CurrentTickVelocity - PreviousTickVelocity) / DeltaTime;
		AngularAcceleration = (CurrentAngularVelocity - PreviousTickAngularVelocity) / DeltaTime;
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("DeltaTime is equal to zero. Acceleration will be the same as it was the last time."));
	}	

	PreviousTickVelocity = CurrentTickVelocity;
	PreviousTickAngularVelocity = CurrentAngularVelocity;

	if(UpdateSplineProgress())
	{
		if (bIsGameDriving && !IsInputEnabled)
		{
			MoveAlongSpline();
		}
	}
}

void ACar::SetControls(float Steering, float Throttle, float Brake, float Handbrake)
{
	SetSteering(Steering);
	SetThrottle(Throttle);
	SetBrake(Brake);
	SetHandbrake(Handbrake != 0.0f);
}

void ACar::SetSteering(float NewSteering)
{
	LastAppliedSteering = NewSteering;
	GetVehicleMovementComponent()->SetSteeringInput(NewSteering);
}

float ACar::GetPhysicsSteering() const
{
	UFloatProperty* FloatProperty = FindField<UFloatProperty>(UWheeledVehicleMovementComponent4W::StaticClass(), TEXT("SteeringInput"));
	uint8* FloatValuePtr = FloatProperty->ContainerPtrToValuePtr<uint8>(GetVehicleMovementComponent());
	float PhysicsSteering = FloatProperty->GetFloatingPointPropertyValue(FloatValuePtr);

	return PhysicsSteering;
}

float ACar::GetInputSteering() const
{
	return LastAppliedSteering;
}

void ACar::SetThrottle(float NewThrottle)
{
	LastAppliedThrottle = NewThrottle;
	GetVehicleMovementComponent()->SetThrottleInput(NewThrottle);
}

float ACar::GetPhysicsThrottle() const
{
	UFloatProperty* FloatProperty = FindField<UFloatProperty>(UWheeledVehicleMovementComponent4W::StaticClass(), TEXT("ThrottleInput"));
	uint8* FloatValuePtr = FloatProperty->ContainerPtrToValuePtr<uint8>(GetVehicleMovementComponent());
	float PhysicsThrottle = FloatProperty->GetFloatingPointPropertyValue(FloatValuePtr);

	return PhysicsThrottle;
}

float ACar::GetInputThrottle() const
{
	return LastAppliedThrottle;
}

void ACar::SetBrake(float NewBrake)
{
	LastAppliedBrake = NewBrake;
	GetVehicleMovementComponent()->SetBrakeInput(NewBrake);
}

float ACar::GetPhysicsBrake() const
{
	UFloatProperty* FloatProperty = FindField<UFloatProperty>(UWheeledVehicleMovementComponent4W::StaticClass(), TEXT("BrakeInput"));
	uint8* FloatValuePtr = FloatProperty->ContainerPtrToValuePtr<uint8>(GetVehicleMovementComponent());
	float PhysicsBrake = FloatProperty->GetFloatingPointPropertyValue(FloatValuePtr);

	return PhysicsBrake;
}

float ACar::GetInputBrake() const
{
	return LastAppliedBrake;
}

void ACar::SetHandbrake(bool NewHandbrake)
{
	LastAppliedHandbrake = NewHandbrake;
	GetVehicleMovementComponent()->SetHandbrakeInput(NewHandbrake);
}

float ACar::GetPhysicsHandbrake() const
{
	UFloatProperty* FloatProperty = FindField<UFloatProperty>(UWheeledVehicleMovementComponent4W::StaticClass(), TEXT("HandbrakeInput"));
	uint8* FloatValuePtr = FloatProperty->ContainerPtrToValuePtr<uint8>(GetVehicleMovementComponent());
	float PhysicsHandbrake = FloatProperty->GetFloatingPointPropertyValue(FloatValuePtr);

	return PhysicsHandbrake;
}

bool ACar::GetInputHandbrake() const
{
	return LastAppliedHandbrake;
}

void ACar::SetIsGameDriving(bool NewbIsGameDriving)
{
	bIsGameDriving = NewbIsGameDriving;
}

bool ACar::GetIsGameDriving() const
{
	return bIsGameDriving;
}

void ACar::SetShouldReset(bool NewbShouldReset)
{
	bShouldReset = NewbShouldReset;
}

bool ACar::GetShouldReset() const
{
	return bShouldReset;
}

void ACar::SetIsResetting(bool NewbIsResetting)
{
	bIsResetting = NewbIsResetting;
}

bool ACar::GetIsResetting() const
{
	return bIsResetting;
}

FVector ACar::GetAngularVelocity() const
{
	if (GetMesh() != nullptr)
	{
		return GetMesh()->GetPhysicsAngularVelocity();
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("GetMesh() is nullptr, can't get angular velocity from it"));
		return FVector(0.f, 0.f, 0.f);
	}
}

FUnrealToPythonDataLoad ACar::GenerateOutputData() const
{
	FUnrealToPythonDataLoad OutputData;
	OutputData.Speed = GetVehicleMovementComponent()->GetForwardSpeed();
	OutputData.Position = GetActorLocation();
	OutputData.Velocity = GetVelocity();

	OutputData.Rotation.X = GetActorRotation().Roll;
	OutputData.Rotation.Y = GetActorRotation().Pitch;
	OutputData.Rotation.Z = GetActorRotation().Yaw;

	FVector Origin, BoxExtent;
	GetActorBounds(true, Origin, BoxExtent);
	OutputData.Dimensions = BoxExtent;
	
	OutputData.Acceleration = GetAcceleration();

	FVector angularVelocity;

	angularVelocity = GetAngularVelocity();
	OutputData.RotationVelocity = angularVelocity;

	OutputData.RotationAcceleration = GetAngularAcceleration();

	OutputData.ForwardVector = GetActorForwardVector();
	OutputData.RightVector = GetActorRightVector();
	OutputData.UpVector = GetActorUpVector();

	OutputData.TransformationMatrix = GetActorTransform().ToMatrixWithScale();
	OutputData.bIsGameDriving = GetIsGameDriving();
	OutputData.bIsResetting = GetIsResetting();

	OutputData.AgentTime = GetAgentTime();
	OutputData.EpisodeStartTime = GetEpisodeStartTime();
	OutputData.EpisodeEndTime = GetEpisodeEndTime();

	// OutputData.AgentTime = Gettim
	return OutputData;
}

FVector ACar::GetAcceleration() const
{
	return Acceleration;
}

FVector ACar::GetAngularAcceleration() const
{
	return AngularAcceleration;
}

double ACar::GetAgentTime() const
{
	float WorldSeconds = GetWorld()->GetTimeSeconds();
	FTimespan TimeSpan = FTimespan(/*Hours*/0,/*Minutes*/0, WorldSeconds /*Seconds*/);
	return TimeSpan.GetTotalMicroseconds();
}

double ACar::GetEpisodeStartTime() const
{
	return EpisodeStartTime;
}

double ACar::GetEpisodeEndTime() const
{
	return -1;
}

bool ACar::UpdateSplineProgress()
{
	if (Trajectory.Get() == nullptr)
	{
		UE_LOG(LogTemp, VeryVerbose, TEXT("getting trajectory"));

		for (TActorIterator<ASplineTrajectory> ActorItr(GetWorld()); ActorItr; ++ActorItr)
		{
			Trajectory = *ActorItr;
		}

		if (Trajectory.Get() == nullptr)
		{
			UE_LOG(LogTemp, VeryVerbose, TEXT("car AI is turned on but no spline is set for movement"));
			return false;
		}
	}
	FVector CurrentLocation = GetActorLocation();
	auto ClosestSplineLocation = Trajectory->Trajectory->FindLocationClosestToWorldLocation(CurrentLocation, ESplineCoordinateSpace::World);
	DistanceToCenterOfLane = sqrt(FVector::DistSquaredXY(CurrentLocation, ClosestSplineLocation));
	GetDistanceAlongRouteAtLocation(CurrentLocation);
	WaypointDistanceAlongSpline = static_cast<int>(DistanceAlongRoute + WaypointStep + CloseDistanceThreshold) / static_cast<int>(WaypointStep) * WaypointStep; // Assumes whole number waypoint step

	UE_LOG(LogTemp, VeryVerbose, TEXT("WaypointDistanceAlongSpline %f"), WaypointDistanceAlongSpline);

	FVector WaypointPosition = Trajectory->Trajectory->GetLocationAtDistanceAlongSpline(WaypointDistanceAlongSpline, ESplineCoordinateSpace::World);
	if (FVector::Dist(WaypointPosition, CurrentLocation) < CloseDistanceThreshold)
	{
		// We've gotten to the next waypoint along the spline
		WaypointDistanceAlongSpline += WaypointStep; // TODO: Remove assumption here that we are travelling at speeds and framerates for this to make sense.
		if (WaypointDistanceAlongSpline > Trajectory->Trajectory->GetSplineLength())
		{
			UE_LOG(LogTemp, Warning, TEXT("resetting target point on spline after making full trip around track"));

			WaypointDistanceAlongSpline = 0.f;
			LapNumber += 1;
		}
	}

	return true;
}

bool ACar::searchAlongSpline(FVector CurrentLocation, int step, float distToCurrent, float& distanceAlongRoute)
{	int i = 0;
	while (true)
	{
		FVector nextLocation = Trajectory->Trajectory->GetLocationAtDistanceAlongSpline(distanceAlongRoute + step, ESplineCoordinateSpace::World);
		float distToNext = FVector::Dist(CurrentLocation, nextLocation);
		UE_LOG(LogTemp, VeryVerbose, TEXT("distToCurrent %f, distToNext %f, distanceAlongRoute %f, step %d"), distToCurrent, distToNext, distanceAlongRoute, step);

		if (distToNext > distToCurrent) // || distanceAlongRoute <= static_cast<float> (abs(step))
		{
			return true;
		}
		else if (i > 9)
		{
			UE_LOG(LogTemp, VeryVerbose, TEXT("searched %d steps for closer spline point - giving up!"), i);
			return false;
		}
		else {
			UE_LOG(LogTemp, VeryVerbose, TEXT("advancing distance along route: %f by step %d"), distanceAlongRoute, step);
			distToCurrent = distToNext;
			distanceAlongRoute += step;
		}
		i++;
	}
}

bool ACar::getDistanceAlongSplineAtLocationWithStep(FVector CurrentLocation, unsigned int step, float& distanceAlongRoute)
{
	// Search the spline, starting from our previous distance, for locations closer to our current position. 
	// Then return the distance associated with that position.
	FVector prevLocation   = Trajectory->Trajectory->GetLocationAtDistanceAlongSpline(distanceAlongRoute,        ESplineCoordinateSpace::World);
	FVector locationAhead  = Trajectory->Trajectory->GetLocationAtDistanceAlongSpline(distanceAlongRoute + step, ESplineCoordinateSpace::World);
	FVector locationBehind = Trajectory->Trajectory->GetLocationAtDistanceAlongSpline(distanceAlongRoute - step, ESplineCoordinateSpace::World);

	float distToPrev   = FVector::Dist(CurrentLocation, prevLocation);
	float distToAhead  = FVector::Dist(CurrentLocation, locationAhead);
	float distToBehind = FVector::Dist(CurrentLocation, locationBehind);
	UE_LOG(LogTemp, VeryVerbose, TEXT("distToPrev: %f, distToAhead: %f, distToBehind %f"), distToPrev, distToAhead, distToBehind);

	bool found = false;

	if (distToAhead <= distToPrev && distToAhead <= distToBehind)
	{
		// Move forward
		UE_LOG(LogTemp, VeryVerbose, TEXT("moving forward"));
		found = searchAlongSpline(CurrentLocation, step, distToAhead, distanceAlongRoute);
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
		found = searchAlongSpline(CurrentLocation, - static_cast<int>(step), distToBehind, distanceAlongRoute);
	}
	else
	{
		UE_LOG(LogTemp, Error, TEXT("Unexpected distance to waypoint! distToPrev: %f, distToAhead: %f, distToBehind %f"), distToPrev, distToAhead, distToBehind);
		// throw std::logic_error("Unexpected distance to waypoint!");
	}
	return found;
}

void ACar::GetDistanceAlongRouteAtLocation(FVector CurrentLocation)
{
	// Search the spline's distance table starting at the last calculated distance: first at 1 meter increments, then 10cm

	//   TODO: Binary search
	if( ! getDistanceAlongSplineAtLocationWithStep(CurrentLocation, 100, DistanceAlongRoute))
	{
		// Our starting point way off base - search with larger steps
		getDistanceAlongSplineAtLocationWithStep(CurrentLocation, 1000, DistanceAlongRoute);

		getDistanceAlongSplineAtLocationWithStep(CurrentLocation, 100, DistanceAlongRoute);
	}
	UE_LOG(LogTemp, VeryVerbose, TEXT("dist 1 meter %f"), DistanceAlongRoute);

	getDistanceAlongSplineAtLocationWithStep(CurrentLocation, 10, DistanceAlongRoute);
	UE_LOG(LogTemp, VeryVerbose, TEXT("dist 100 cm %f"), DistanceAlongRoute);
}

void ACar::MoveAlongSpline()
{
	// TODO: Poor man's motion planning - Look ahead n-steps and check tangential difference in yaw, pitch, and roll - then reduce velocity accordingly
	// TODO: 3D motion planning

	FVector CurrentLocation2 = GetActorLocation();
	FVector CurrentTarget = Trajectory->Trajectory->GetLocationAtDistanceAlongSpline(WaypointDistanceAlongSpline, ESplineCoordinateSpace::World);

	float CurrentYaw = GetActorRotation().Yaw;
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

	SetSteering(YawDifference / 30.f);

	auto speed = GetVehicleMovementComponent()->GetForwardSpeed();

	if(speed > 1800)
	{
		AIThrottle = 0;
	}
	else if(speed < 1600)
	{
		AIThrottle = 1.0f;
	}
	else
	{
		AIThrottle = 0.75f;
	}

	/*if (FMath::Abs(YawDifference) < ZeroSteeringToleranceDeg)
	{
		SetSteering(0);
	}
	else
	{
		SetSteering(FMath::Sign(YawDifference));
	}*/

	SetThrottle(AIThrottle);
}
