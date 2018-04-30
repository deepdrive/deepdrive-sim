
#pragma once

#include "Engine.h"


UENUM(BlueprintType)
enum class EDeepDriveAgentControlMode : uint8
{
	NONE			= 0	UMETA(DisplayName = "None"),
	MANUAL			= 1	UMETA(DisplayName = "Manual"),
	SPLINE			= 2	UMETA(DisplayName = "Spline"),
	REMOTE_AI		= 3	UMETA(DisplayName = "RemoteAI"),
	LOCAL_AI		= 4	UMETA(DisplayName = "LocalAI")
};


UENUM(BlueprintType)
enum class EDeepDriveAgentCameraType : uint8
{
	NONE				= 0	UMETA(DisplayName = "None"),
	CHASE_CAMERA		= 1	UMETA(DisplayName = "ChaseCamera"),
	INTERIOR_CAMERA		= 2	UMETA(DisplayName = "InteriorCamera"),
	ORBIT_CAMERA	    = 3	UMETA(DisplayName = "OrbitCameraCamera"),
	FREE_CAMERA			= 4	UMETA(DisplayName = "FreeCamera")
};

