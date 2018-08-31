
#pragma once

#include "Engine.h"

struct DeepDriveCollisionData
{
	FDateTime	LastCollisionTimeUTC = FDateTime::FromUnixTimestamp(0);

	double		LastCollisionTimeStamp = 0.0;

	double		TimeSinceLastCollision = 0.0;

	FName		CollisionLocation;

	FVector		ColliderVelocity = FVector(0.0f, 0.0f, 0.0f);

	FVector		CollisionNormal = FVector(0.0f, 0.0f, 0.0f);
};


struct DeepDriveDataOut
{
	FVector		Position;

	FRotator	Rotation;

	FVector		Velocity;

	FVector		Acceleration;

	FVector		AngularVelocity;

	FVector		AngularAcceleration;

	FVector		Dimension;

	float		Speed;

	bool		IsGameDriving;

	bool		IsResetting;

	float		Steering;

	float		Throttle;

	float		Brake;

	bool		Handbrake;

	float		DistanceAlongRoute;

	float		DistanceToCenterOfLane;

	int			LapNumber;

	DeepDriveCollisionData		CollisionData;
};
