
#pragma once

#include "Engine.h"
#include "Runtime/CoreUObject/Public/UObject/ObjectMacros.h"

#include "DeepDriveSimulationDefines.generated.h"

class UDeepDriveRandomStream;
class ADeepDriveAgent;

UENUM(BlueprintType)
enum class EDeepDriveAgentControlMode : uint8
{
	NONE			= 0	UMETA(DisplayName = "None"),
	MANUAL			= 1	UMETA(DisplayName = "Manual"),
	REMOTE_AI		= 2	UMETA(DisplayName = "RemoteAI"),
	LOCAL_AI		= 3	UMETA(DisplayName = "LocalAI")
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


USTRUCT(BlueprintType)
struct FDeepDriveRandomStreamData
{
	GENERATED_USTRUCT_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	bool	ReSeedOnReset = true;

	FDeepDriveRandomStreamData()
		:	ReSeedOnReset(true)
		,	RandomStream(0)
	{	}

	FDeepDriveRandomStreamData(UDeepDriveRandomStream *randomStream, bool reseedOnReset)
		:	ReSeedOnReset(reseedOnReset)
		,	RandomStream(randomStream)
	{	}

	void setRandomStream(UDeepDriveRandomStream *randomStream)
	{
		RandomStream = randomStream;
	}

	UDeepDriveRandomStream* getRandomStream()
	{
		return RandomStream;
	}

private:

	UPROPERTY()
	UDeepDriveRandomStream		*RandomStream = 0;
};

USTRUCT(BlueprintType)
struct FDeepDriveAdditionalAgentData
{
	GENERATED_USTRUCT_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	TSubclassOf<ADeepDriveAgent>	Agent;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	EDeepDriveAgentControlMode		Mode;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	int32	ConfigurationSlot;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	int32	StartPositionSlot;
};


USTRUCT(BlueprintType)
struct FDeepDriveViewMode
{
	GENERATED_USTRUCT_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	UMaterialInterface	*Material = 0;
};
