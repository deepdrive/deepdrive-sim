

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "DeepDriveTrafficLight.generated.h"


UENUM(BlueprintType)
enum class EDeepDriveTrafficLightPhase : uint8
{
	RED 				= 0	UMETA(DisplayName = "Red"),
	RED_TO_GREEN 		= 1	UMETA(DisplayName = "Red to Green"),
	GREEN 				= 2	UMETA(DisplayName = "Green"),
	GREEN_TO_RED    	= 3	UMETA(DisplayName = "Green to Red")
};



UCLASS()
class DEEPDRIVEPLUGIN_API ADeepDriveTrafficLight : public AActor
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	ADeepDriveTrafficLight();

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

	UFUNCTION(BlueprintImplementableEvent, Category = "Control")
	void OnPhaseChanged();

	UPROPERTY(BlueprintReadOnly, Category = Phase)
	EDeepDriveTrafficLightPhase		CurrentPhase = EDeepDriveTrafficLightPhase::RED;

	UPROPERTY(EditAnywhere, Category = Configuration)
	float	RedToGreenDuration = 2.0f;

	UPROPERTY(EditAnywhere, Category = Configuration)
	float	GreenToRedDuration = 3.0f;

	void SwitchToGreen();

	void SwitchToRed();

	void SetToGreen(float ElapsedTime);

	void SetToRed(float ElapsedTime);

	void Deactivate();

	void TurnOff();

private:

	float								m_remainingPhaseTime = -1.0f;
	EDeepDriveTrafficLightPhase			m_nextPhase = EDeepDriveTrafficLightPhase::RED;

};
