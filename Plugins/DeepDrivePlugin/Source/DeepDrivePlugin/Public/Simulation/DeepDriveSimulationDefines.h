
#pragma once

#include "Engine.h"
#include "Runtime/CoreUObject/Public/UObject/ObjectMacros.h"

#include "DeepDriveSimulationDefines.generated.h"

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

USTRUCT(BlueprintType)
struct FDeepDriveControllerData
{
	GENERATED_USTRUCT_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	float	FloatParameter1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	float	FloatParameter2;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	float	FloatParameter3;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	float	FloatParameter4;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	AActor	*Actor1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	AActor	*Actor2;

};

