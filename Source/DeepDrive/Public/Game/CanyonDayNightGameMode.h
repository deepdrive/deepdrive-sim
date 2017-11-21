// SDP TEAM

#pragma once

#include "GameFramework/GameModeBase.h"
#include "CanyonDayNightGameMode.generated.h"

/**
 * 
 */
UCLASS()
class DEEPDRIVE_API ACanyonDayNightGameMode : public AGameMode
{
	GENERATED_BODY()
	
public:

	ACanyonDayNightGameMode();

	virtual void BeginPlay() override;

	virtual void Tick(float DeltaTime) override;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "DayNight")
	float DayNightCycleSec;


	TSet<class AStaticMeshActor*> ConvertedToDynamic;

	TArray<class UMaterialInstanceDynamic*> BuildingDynamicMaterials;

	TArray<float> BuildingDynamicThresholds;

	TArray<class UMaterialInstanceDynamic*> SkyDynamicMaterials;
	
	
};
