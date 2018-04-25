
#pragma once

#include "SunLightSimulatorComponent.generated.h"

DECLARE_LOG_CATEGORY_EXTERN(LogSunLightSimulator, Log, All);

UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent) )
class DEEPDRIVEPLUGIN_API USunLightSimulatorComponent : public UActorComponent
{
	GENERATED_BODY()

public:

	UFUNCTION(BlueprintCallable, Category = "Texture")
	UTexture2D* CreateLookUpTexture(int32 Width, int32 NumSamples);

	UFUNCTION(BlueprintCallable, Category = "SunLight")
	FVector CalculateSunLightColor(const FVector &SunDirection);

private:

	void setupTable(int32 width, int32 numSamples);

	void calculateOpticalDepth(float height, const FVector &dir, int32 numSamples, float &odRayleigh, float &odMie);

	float intersect(const FVector &start, const FVector &dir);

	TArray<float>			m_RayleighTable;
	TArray<float>			m_MieTable;

	const float				m_Hr = 7.994f;			// Rayleigh scale height
	const float				m_Hm = 1.200f;			// Mie scale height
	const float				m_EarthRadius = 6360;
	const float				m_AtmosphereRadius = 6519;

	const FVector			m_BetaR = FVector(5.5e-6, 13.0e-6, 22.4e-6);
	const FVector			m_BetaM = FVector(21e-6, 21e-6, 21e-6);


	const float				m_RefHeight = 1.59f;

};
