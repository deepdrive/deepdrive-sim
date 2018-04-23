
#pragma once

#include "SunLightSimulatorComponent.generated.h"

UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent) )
class DEEPDRIVEPLUGIN_API USunLightSimulatorComponent : public UActorComponent
{
	GENERATED_BODY()

public:

	UFUNCTION(BlueprintCallable, Category = "Texture")
	UTexture2D* CreateLookUpTexture(int32 Width, int32 NumSamples);

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

};
