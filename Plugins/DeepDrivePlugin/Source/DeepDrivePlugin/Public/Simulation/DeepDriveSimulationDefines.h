
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
	CHASE_CAMERA		= 0	UMETA(DisplayName="ChaseCamera"),
	INTERIOR_CAMERA		= 1	UMETA(DisplayName="InteriorCamera"),
	ORBIT_CAMERA	    = 2	UMETA(DisplayName="OrbitCameraCamera"),
	FREE_CAMERA			= 3	UMETA(DisplayName="FreeCamera")
};

