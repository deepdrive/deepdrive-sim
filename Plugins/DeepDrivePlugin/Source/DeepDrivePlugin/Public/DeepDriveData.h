
#pragma once

#include "Engine.h"

struct DeepDriveCollisionData
{
	DeepDriveCollisionData()
		: LastCollisionTimeUTC()
		, LastCollisionTimeStamp(-1.0)
		, TimeSinceLastCollision(-1.0)
		, CollisionLocation()
		, ColliderVelocity()
		, CollisionNormal()
	{
	}

	FDateTime	LastCollisionTimeUTC;

	double		LastCollisionTimeStamp;

	double		TimeSinceLastCollision;

	FName		CollisionLocation;

	FVector		ColliderVelocity;

	FVector		CollisionNormal;
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
