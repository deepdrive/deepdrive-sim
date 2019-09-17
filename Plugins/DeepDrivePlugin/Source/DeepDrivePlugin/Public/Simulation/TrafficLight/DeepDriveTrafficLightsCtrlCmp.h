

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "DeepDriveTrafficLightsCtrlCmp.generated.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveTrafficLightsCtrlCmp, Log, All);

class ADeepDriveTrafficLight;

USTRUCT(BlueprintType)
struct FDeepDriveTrafficLightCircuit
{
	GENERATED_USTRUCT_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	float								Delay;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	float								Duration;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	TArray<ADeepDriveTrafficLight*>		TrafficLights;

	void resetState()
	{
		State = -1;
	}

	void setState(bool newState)
	{
		State = newState;
	}

	int8 getState() const
	{
		return State;
	}

private:

	float								TotalDuration = 0.0f;
	int8								State;		//	< 0 -> Waiting for green, == 0 -> green, > 0 green expired
};

USTRUCT(BlueprintType)
struct FDeepDriveTrafficLightCycle
{
	GENERATED_USTRUCT_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	float									Duration;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	TArray<FDeepDriveTrafficLightCircuit>	Circuits;
};

UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent) )
class DEEPDRIVEPLUGIN_API UDeepDriveTrafficLightsCtrlCmp : public UActorComponent
{
	GENERATED_BODY()

public:	
	// Sets default values for this component's properties
	UDeepDriveTrafficLightsCtrlCmp();

protected:
	// Called when the game starts
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	TArray<FDeepDriveTrafficLightCycle>				Cycles;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	float	TimeOffset = 0.0f;

private:

	float							m_curCycleTime = 0.0f;

	int32							m_curCycleIndex = 0;

};
