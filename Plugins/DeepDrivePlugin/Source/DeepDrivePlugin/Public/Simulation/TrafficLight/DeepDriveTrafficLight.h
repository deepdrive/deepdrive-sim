

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "DeepDriveTrafficLight.generated.h"


UENUM(BlueprintType)
enum class EDeepDriveTrafficLightPhase : uint8
{
	OFF 				= 0	UMETA(DisplayName = "Off"),
	INACTIVE			= 1	UMETA(DisplayName = "Inactive"),
	RED 				= 2	UMETA(DisplayName = "Red"),
	RED_TO_GREEN 		= 3	UMETA(DisplayName = "Red to Green"),
	GREEN 				= 4	UMETA(DisplayName = "Green"),
	GREEN_TO_RED    	= 5	UMETA(DisplayName = "Green to Red")
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

	bool IsActive() const;

private:

	float								m_remainingPhaseTime = -1.0f;
	EDeepDriveTrafficLightPhase			m_nextPhase = EDeepDriveTrafficLightPhase::RED;

};

inline bool ADeepDriveTrafficLight::IsActive() const
{
	return CurrentPhase != EDeepDriveTrafficLightPhase::OFF && CurrentPhase != EDeepDriveTrafficLightPhase::INACTIVE;
}
