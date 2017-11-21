
#pragma once

#include "Engine.h"

UENUM(BlueprintType)
enum class EDeepDriveCameraType : uint8
{
	DDC_CAMERA_NONE			= 0	UMETA(DisplayName="None"),
	DDC_CAMERA_FRONT		= 1	UMETA(DisplayName="FrontCamera"),
	DDC_CAMERA_LEFT		    = 2	UMETA(DisplayName="LeftCamera"),
	DDC_CAMERA_RIGHT		= 3	UMETA(DisplayName="RightCamera"),
	DDC_CAMERA_FRONT_LEFT	= 4	UMETA(DisplayName="FrontLeftCamera"),
	DDC_CAMERA_FRONT_RIGHT	= 5	UMETA(DisplayName="FrontRightCamera"),
	DDC_CAMERA_BACK_LEFT	= 6	UMETA(DisplayName="BackLeftCamera"),
	DDC_CAMERA_BACK_RIGHT	= 7	UMETA(DisplayName="BackRightCamera"),
	DDC_CAMERA_BACK			= 8	UMETA(DisplayName="BackCamera")
};
