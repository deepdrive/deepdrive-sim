
#pragma once

#include "Engine.h"

/*
	Unreal -> (Python) Client
*/

struct FDeepDriveDataOut
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

	FDateTime	LastCollisionTimeUTC;

	double		LastCollisionTimeStamp;

	double		TimeSinceLastCollision;

};
