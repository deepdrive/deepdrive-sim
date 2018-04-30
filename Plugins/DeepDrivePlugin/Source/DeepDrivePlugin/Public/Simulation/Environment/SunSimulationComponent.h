
#pragma once

#include "SunSimulationComponent.generated.h"

DECLARE_LOG_CATEGORY_EXTERN(LogSunSimulationComponent, Log, All);

UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent) )
class DEEPDRIVEPLUGIN_API USunSimulationComponent : public UActorComponent
{
	GENERATED_BODY()

public:

	UFUNCTION(BlueprintCallable, Category = "SunSimulation")
	void Advance(float DeltaSeconds);

	UFUNCTION(BlueprintCallable, Category = "Texture")
	UTexture2D* CreateLookUpTexture(int32 Width, int32 NumSamples);

	UFUNCTION(BlueprintCallable, Category = "SunLight")
	FLinearColor CalculateSunLightColor(const FVector &SunDirection, float Radius, int32 NumSamples);

	UPROPERTY(BlueprintReadOnly, Category = "SunPosition")
	float	Azimuth = 0.0f;

	UPROPERTY(BlueprintReadOnly, Category = "SunPosition")
	float	Zenith = 0.0f;

	UFUNCTION(BlueprintCallable, Category = "SunSimulation")
	void SetLocation(float LongitudeDegrees, float LatitudeDegrees);

	UFUNCTION(BlueprintCallable, Category = "SunSimulation")
	void SetDate(int32 Year, int32 Month, int32 Day, float Duration);

	UFUNCTION(BlueprintCallable, Category = "SunSimulation")
	void SetTime(int32 Hour, int32 Minute, float Duration);

	UFUNCTION(BlueprintCallable, Category = "SunSimulation")
	void SetDateAndTime(int32 Year, int32 Month, int32 Day, int32 Hour, int32 Minute, float Duration);

	UFUNCTION(BlueprintCallable, Category = "SunSimulation")
	void EnableSimulation(bool Enabled);

	UFUNCTION(BlueprintCallable, Category = "SunSimulation")
	void SetSimulationSpeed(int32 MinutesPerSecond);

private:

	void setupTable(int32 width, int32 numSamples);

	FLinearColor calcSunLightColor(const FVector &sunDirection);

	void calculateOpticalDepth(float height, const FVector &dir, int32 numSamples, float &odRayleigh, float &odMie);

	float intersect(const FVector &start, const FVector &dir);

	void advanceTime(float DeltaSeconds);

	void prepareTimespan(float duration);

	void calculatePosition();

	double getJulianDay();
	double getJulianDate(double julianDay);

	double normalizeAngleDeg(double angle);

	double deg2rad(double angle);
	double rad2deg(double angle);

	TArray<float>			m_RayleighTable;
	TArray<float>			m_MieTable;

	const float				m_Hr = 7.994f;			// Rayleigh scale height
	const float				m_Hm = 1.200f;			// Mie scale height
	const float				m_EarthRadius = 6360;
	const float				m_AtmosphereRadius = 6519;

	const FVector			m_BetaR = FVector(5.5e-6, 13.0e-6, 22.4e-6);
	const FVector			m_BetaM = FVector(21e-6, 21e-6, 21e-6);

	const float				m_RefHeight = 1.59f;

	float					m_LongitudeDegrees = 9;
	float					m_LatitudeDegrees = 53;

	double					m_JulianDay = 0;
	FDateTime				m_curDateTime;
	FDateTime				m_desiredDateTime;
	FTimespan				m_curTimeSpan = FTimespan::Zero();

	int32					m_MinutesPerSecond = 0;

	enum SimulationMode
	{
		Idle,
		Forward,
		Continous
	};

	SimulationMode			m_SimulationMode = Idle;

};


inline double USunSimulationComponent::deg2rad(double angle)
{
	return angle * PI / 180.0;
}

inline double USunSimulationComponent::rad2deg(double angle)
{
	return angle * 180.0 / PI;
}
