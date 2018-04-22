
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

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "Simulation")
	float	LongitudeDegrees = 9;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "Simulation")
	float	LatitudeDegrees = 53;

	UPROPERTY(BlueprintReadWrite, Category = "Simulation")
	int32	MinutesPerSecond = 0;

	UFUNCTION(BlueprintCallable, Category = "Simulation")
	void SetDate(int32 Year, int32 Month, int32 Day, int32 Hour);

	UFUNCTION(BlueprintCallable, Category = "Simulation")
	void Advance(float DeltaSeconds);

private:

	void forward(float timeDelta);

	void advancePSA(float DeltaSeonds);
	void advanceClassic(float DeltaSeonds);

	double getJulianDay();
	double getJulianDate(double julianDay);

	double normalizeAngleDeg(double angle);

	double deg2rad(double angle);
	double rad2deg(double angle);

	float				m_AzimuthOffset = 0.0f;

	double				m_JulianDay = 0;

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
