
#pragma once

#include "Engine.h"


UENUM(BlueprintType)
enum class EDeepDriveAgentControlMode : uint8
{
	DDACM_NONE			= 0	UMETA(DisplayName = "None"),
	DDACM_MANUAL		= 1	UMETA(DisplayName = "Manual"),
	DDACM_SPLINE		= 2	UMETA(DisplayName = "Spline"),
	DDACM_REMOTE_AI		= 3	UMETA(DisplayName = "RemoteAI"),
	DDACM_LOCAL_AI		= 4	UMETA(DisplayName = "LocalAI")
};


UENUM(BlueprintType)
enum class EDeepDriveAgentCameraType : uint8
{
	DDAC_CHASE_CAMERA		= 0	UMETA(DisplayName="ChaseCamera"),
	DDAC_INTERIOR_CAMERA	= 1	UMETA(DisplayName="InteriorCamera"),
	DDAC_ORBIT_CAMERA	    = 2	UMETA(DisplayName="OrbitCameraCamera"),
	DDAC_FREE_CAMERA		= 3	UMETA(DisplayName="FreeCamera")
};

