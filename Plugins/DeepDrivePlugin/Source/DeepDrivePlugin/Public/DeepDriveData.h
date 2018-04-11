
#pragma once

#include "DeepDriveData.generated.h"

/*
	Unreal -> (Python) Client
*/

USTRUCT(BlueprintType)
struct FDeepDriveDataOut
{
	GENERATED_USTRUCT_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	FVector		Position;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	FRotator	Rotation;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	FVector		Velocity;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	FVector		Acceleration;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	FVector		AngularVelocity;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	FVector		AngularAcceleration;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	FVector		Dimension;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	float		Speed;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	bool		IsGameDriving;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	bool		IsResetting;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	float		Steering;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	float		Throttle;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	float		Brake;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	bool		Handbrake;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	float		DistanceAlongRoute;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	float		DistanceToCenterOfLane;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	int			LapNumber;

};

USTRUCT(BlueprintType)
struct FDeepDriveControlData
{
	GENERATED_USTRUCT_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	float		Steering;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	float		Throttle;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	float		Brake;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	float		Handbrake;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	bool		IsGameDriving;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	bool		ShouldReset;

};
