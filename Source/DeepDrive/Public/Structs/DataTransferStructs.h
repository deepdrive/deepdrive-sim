// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "DataTransferStructs.generated.h"


// lets evade epic's legacy bool serializator
FORCEINLINE void SaveLoadBoolUint8(FArchive& Ar, bool& D)
{
	if (Ar.IsLoading())
	{
		uint8 Value;
		Ar << Value;
		D = Value > 0;
	}
	else
	{
		uint8 Value = D ? 1 : 0;
		Ar << Value;
	}
}

USTRUCT(Blueprintable)
struct FOutputCameraData
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadWrite)
	int32 Width;

	UPROPERTY(BlueprintReadWrite)
	int32 Height;

	// 16 bit per channel
	UPROPERTY(BlueprintReadWrite)
	TArray<uint8> RGBArray;

	UPROPERTY(BlueprintReadWrite)
	float FOV;

	// can be calculated from Width/Height
	// UPROPERTY(BlueprintReadWrite)
	// float AspectRatio;

	UPROPERTY(BlueprintReadWrite)
	FMatrix CameraMatrix;

	// enumeration?
	UPROPERTY(BlueprintReadWrite)
	FString CameraTypeName;

	// enumeration?
	UPROPERTY(BlueprintReadWrite)
	int32 CameraId;
};


USTRUCT(Blueprintable)
struct FUnrealToPythonDataLoad
{
	GENERATED_BODY()

	// HDR / depth RGB in 16bit per channel for all cameras + fov, aspect ratio, and camera intrinsics / camera extrinsics matrix, camera_type_name, id
	UPROPERTY(BlueprintReadWrite)
	TArray<FOutputCameraData> CameraDataArray;

	// speed along forward vector of vehicle, what the spedometer would tell you, cm/s
	UPROPERTY(BlueprintReadWrite)
	float Speed;

	// x,y,z from origin coordinates of ego vehicle, cm
	UPROPERTY(BlueprintReadWrite)
	FVector Position;

	// x,y,z velocity of vehicle in frame of ego origin orientation, cm/s
	UPROPERTY(BlueprintReadWrite)
	FVector Velocity;

	// roll, pitch, yaw of vehicle, in degrees
	UPROPERTY(BlueprintReadWrite)
	FVector Rotation;

	// height, width, length of ego in meters
	UPROPERTY(BlueprintReadWrite)
	FVector Dimensions;

	// frictional coefficient of road surface, release 2
	UPROPERTY(BlueprintReadWrite)
	float Release2_SurfaceTraction;

	// x,y,z acceleration
	UPROPERTY(BlueprintReadWrite)
	FVector Acceleration;

	// roll, pitch, yaw rotation velocity
	UPROPERTY(BlueprintReadWrite)
	FVector RotationVelocity;

	// roll, pitch, yaw rotation acceleration
	UPROPERTY(BlueprintReadWrite)
	FVector RotationAcceleration;

	// unit vector pointing in the forward direction of the car
	UPROPERTY(BlueprintReadWrite)
	FVector ForwardVector;

	// unit vector pointing in the right direction of the car
	UPROPERTY(BlueprintReadWrite)
	FVector UpVector;

	// unit vector pointing in the right direction of the car
	UPROPERTY(BlueprintReadWrite)
	FVector RightVector;

	// 3d transformation matrix FMatrix https://docs.unrealengine.com/latest/INT/API/Runtime/Core/Math/FMatrix/index.html - Unreal coordinates are fine x forward, y left, z up
	UPROPERTY(BlueprintReadWrite)
	FMatrix TransformationMatrix;

	// max number of passengers in vehicle, release 2
	UPROPERTY(BlueprintReadWrite)
	int32 Release2_MaxNumberOfPassengers;

	// number of passengers in vehicle, release 2
	UPROPERTY(BlueprintReadWrite)
	int32 Release2_NumberOfPassengers;

	// reinforcement learning / driving safety, speed, comfort reward, release 2
	UPROPERTY(BlueprintReadWrite)
	float Release2_Reward;

	// whether vehicle is on the road, release 2
	UPROPERTY(BlueprintReadWrite)
	bool Release2_bOnRoad;

	// tire starting clockwise from driver front, release 2
	UPROPERTY(BlueprintReadWrite)
	TArray<bool> Release2_TireHasBurst;

	// g-force seconds past some threshold g-force - forward, lateral, and vertical relative to ego, release 2
	UPROPERTY(BlueprintReadWrite)
	FVector Release2_Comfort;

	// vehicle is airborn, release 2
	UPROPERTY(BlueprintReadWrite)
	bool Release2_bInAir;

	// whether the in-game AI is driving or not
	UPROPERTY(BlueprintReadWrite)
	bool bIsGameDriving;

	UPROPERTY(BlueprintReadWrite)
	bool bIsResetting;

	// visible pedestrians, vehicles, bicycles, motorcycles, animals 
	// id, position, rotation, height, width, length, velocity, rotaional velocity, acceleration, type, up_vector, forward_vector, right_vector, transformation matrix, number_of_passengers, release 2
	// UPROPERTY(BlueprintReadWrite)
	// TArray<FMovingObject> MovingObjects;

	// utc timestamp of last collision, release 2
	UPROPERTY(BlueprintReadWrite)
	int32 Release2_LastCollisionTime;

	// utc timestamp of last time drove in the wrong lane of traffic, release 2
	UPROPERTY(BlueprintReadWrite)
	int32 Release2_LastDroveAgainstTrafficTime;

	// utc timestamp of last time we drove off the road, release 2
	UPROPERTY(BlueprintReadWrite)
	int32 Release2_LastDroveOffroadTime;

	// utc timestamp of current game time microseconds if possible - physics time not system time
	double AgentTime;

	// utc timestamp of start game/episode time microseconds if possible
	double EpisodeStartTime;

	// utc timestamp of end game/episode time microseconds if possible, -1 for not ended
	double EpisodeEndTime;

	// meters along path to destination
	UPROPERTY(BlueprintReadWrite)
	float DistanceAlongPath;

	// meters from left lane line, release 2
	UPROPERTY(BlueprintReadWrite)
	float Release2_DistanceFromLeftLane;

	// meters from right lane line, release 2
	UPROPERTY(BlueprintReadWrite)
	float Release2_DistanceFromRightLane;

	void SaveLoad(FArchive& Ar)
	{
	}
};
