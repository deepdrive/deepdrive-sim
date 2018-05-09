// SDP TEAM

#include "CanyonDayNightGameMode.h"
#include "DeepDrive.h"
#include "Kismet/KismetMathLibrary.h"




ACanyonDayNightGameMode::ACanyonDayNightGameMode()
{
	DayNightCycleSec = 5.f;
}

void ACanyonDayNightGameMode::BeginPlay()
{
	Super::BeginPlay();
}

void ACanyonDayNightGameMode::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	// float DayTime;
	// UKismetMathLibrary::FMod(GetWorld()->GetTimeSeconds(), DayNightCycleSec, DayTime);

	float CycleTime = FMath::Abs(FMath::Sin(GetWorld()->GetTimeSeconds() / DayNightCycleSec * PI));
	// DayTime /= DayNightCycleSec;

	const FName BuildingMaterialParameterName = FName(TEXT("DayTimeMultiplier"));
	const FName SkyMaterialParameterName = FName(TEXT("DayLightLerp"));	

	for (TActorIterator<AStaticMeshActor> ActorItr(GetWorld()); ActorItr; ++ActorItr)
	{
		AStaticMeshActor *Mesh = *ActorItr;
		if (!ConvertedToDynamic.Contains(Mesh))
		{
			auto StaticMeshComponent = Mesh->GetStaticMeshComponent();
			for (int MaterialIndex = 0; MaterialIndex < StaticMeshComponent->GetNumMaterials(); ++MaterialIndex)
			{
				auto Material = StaticMeshComponent->GetMaterial(MaterialIndex);
				if (Material == nullptr)
				{
					continue;
				}

				float UnusedValue;
				
				if (Material->GetScalarParameterValue(BuildingMaterialParameterName, UnusedValue))
				{
					BuildingDynamicMaterials.Add(StaticMeshComponent->CreateDynamicMaterialInstance(MaterialIndex));
					BuildingDynamicThresholds.Add(FMath::FRandRange(0.4, 0.7));
				}

				if (Material->GetScalarParameterValue(SkyMaterialParameterName, UnusedValue))
				{
					SkyDynamicMaterials.Add(StaticMeshComponent->CreateDynamicMaterialInstance(MaterialIndex));
				}
			}		
				
			ConvertedToDynamic.Add(Mesh);
		}
	}

	for (int MaterialIndex = 0; MaterialIndex < BuildingDynamicMaterials.Num(); ++MaterialIndex)
	{
		auto Material = BuildingDynamicMaterials[MaterialIndex];
		float DayTimeMultiplier = CycleTime > BuildingDynamicThresholds[MaterialIndex] ? 1.f : 0.f;
		Material->SetScalarParameterValue(BuildingMaterialParameterName, DayTimeMultiplier);
	}

	for (auto Material : SkyDynamicMaterials)
	{
		Material->SetScalarParameterValue(SkyMaterialParameterName, CycleTime);
	}

	for (TActorIterator<ADirectionalLight> ActorItr(GetWorld()); ActorItr; ++ActorItr)
	{
		auto LightActor = *ActorItr;

		LightActor->SetActorRotation(FRotator(-109.f, GetWorld()->GetTimeSeconds() / DayNightCycleSec * 180.f + 270.f, 150.f));
	}

	for (TActorIterator<ASpotLight> ActorItr(GetWorld()); ActorItr; ++ActorItr)
	{
		auto SpotLightActor = *ActorItr;

		SpotLightActor->SetEnabled(CycleTime > 0.5);
	}
}
