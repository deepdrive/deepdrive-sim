
#pragma once

#include "SunPositionSimulatorComponent.generated.h"

UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent) )
class DEEPDRIVEPLUGIN_API USunPositionSimulatorComponent : public UActorComponent
{
	GENERATED_BODY()

public:

	UPROPERTY(BlueprintReadOnly, Category = "Simulation")
	FVector  SunDirection;

	UPROPERTY(BlueprintReadOnly, Category = "Simulation")
	float	Azimuth = 0.0f;

	UPROPERTY(BlueprintReadOnly, Category = "Simulation")
	float	Zenith = 0.0f;

	UPROPERTY(BlueprintReadWrite, Category = "Simulation")
	int32	MinutesPerSecond = 0;

	UFUNCTION(BlueprintCallable, Category = "Simulation")
	void SetDate(int32 Year, int32 Month, int32 Day, int32 Hour);

	UFUNCTION(BlueprintCallable, Category = "Simulation")
	void Advance(float DeltaSeconds);

private:

	void forward(float timeDelta);

	uint32 getJulianDay();
	double getJulianDate(uint32 julianDay);

	double normalizeAngleDeg(double angle);

	double deg2rad(double angle);
	double rad2deg(double angle);

	float				m_AzimuthOffset = 0.0f;


	double				m_LonDeg = 11.0;
	double				m_LatDeg = 53.0;
	uint32				m_JulianDay = 0;

	FDateTime			m_DateTime;

};

inline double USunPositionSimulatorComponent::deg2rad(double angle)
{
	return angle * PI / 180.0;
}

inline double USunPositionSimulatorComponent::rad2deg(double angle)
{
	return angle * 180.0 / PI;
}
