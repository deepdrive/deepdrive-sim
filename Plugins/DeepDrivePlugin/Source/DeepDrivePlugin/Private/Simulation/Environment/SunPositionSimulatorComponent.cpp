// Fill out your copyright notice in the Description page of Project Settings.

#include "DeepDrivePluginPrivatePCH.h"

#include "Public/Simulation/Environment/SunPositionSimulatorComponent.h"


void USunPositionSimulatorComponent::SetDate(int32 year, int32 month, int32 day, int32 hour)
{
	month = FMath::Clamp<uint16>(month, 1, 12);
	day = FMath::Clamp<uint16>(day, 1, 31);
	hour = FMath::Clamp<uint16>(hour, 0, 23);

	m_DateTime = FDateTime(year, month, day, hour, 0, 0, 0);

	m_JulianDay = getJulianDay();
}


uint32 USunPositionSimulatorComponent::getJulianDay()
{
	// see http://en.wikipedia.org/wiki/Julian_day#Converting_Julian_or_Gregorian_calendar_date_to_Julian_Day_Number
	uint32 a = (14 - m_DateTime.GetMonth()) / 12;
	uint32 y = m_DateTime.GetYear() + 4800 - a;
	uint32 m = m_DateTime.GetMonth() + (12 * a) - 3;

	uint32 julianDay = m_DateTime.GetDay();
	julianDay += ((153 * m) + 2) / 5;
	julianDay += 365 * y;
	julianDay += y / 4;
	julianDay -= y / 100;
	julianDay += y / 400;
	julianDay -= 32045;

	return julianDay;
}

double USunPositionSimulatorComponent::getJulianDate(uint32 julianDay)
{
	double julianDate = julianDay;
	julianDate += (m_DateTime.GetHour() - 12.0) / 24.0;
	julianDate += (m_DateTime.GetMinute()) / 1440.0;
	return julianDate;
}

void USunPositionSimulatorComponent::forward(float minutes)
{
	int32 seconds = 0;
	FTimespan timespan(0, minutes, seconds);
	m_DateTime = m_DateTime + timespan;
}

void USunPositionSimulatorComponent::Advance(float DeltaSeconds)
{
	forward(MinutesPerSecond * DeltaSeconds);

	double julianDate = getJulianDate(m_JulianDay);
	double daysSinceGreenwichNoon = julianDate - 2451545.0;
	double meanLongitude = normalizeAngleDeg(280.460 + (0.9856474 * daysSinceGreenwichNoon));

	double g = deg2rad(normalizeAngleDeg(357.528 + (0.9856003 * daysSinceGreenwichNoon)));
	double ecplipticLongitude = normalizeAngleDeg(meanLongitude + (1.915 * sin(g)) + (0.020 * sin(2 * g)));

	double ecliptic = 23.439 - (0.0000004 * daysSinceGreenwichNoon);
	double epsilon = deg2rad(ecliptic);
	double lambda = deg2rad(ecplipticLongitude);
	double innerDenuminator = cos(lambda);
	double rightAscension = rad2deg(atan(cos(epsilon) * tan(lambda)));
	if (rightAscension != rightAscension)
		return;

	if (innerDenuminator < 0.0)
		rightAscension += 180.0;

	double intPart = 0.0;
	double meanStarTime = modf((6.697376 + (2400.05134 * ((getJulianDate(julianDate) - 2451545.0) / 36525.0)) + (1.002738 * (m_DateTime.GetHour() + ((m_DateTime.GetMinute()) / 60.0)))) / 24.0, &intPart);

	double tau = deg2rad((meanStarTime * 15) + m_LonDeg - rightAscension);
	double phi = deg2rad(m_LatDeg);
	double delta = asin(sin(epsilon) * sin(lambda));
	innerDenuminator = (cos(tau) * sin(phi)) - (tan(delta) * cos(phi));
	double azimuth = atan(sin(tau) / innerDenuminator);
	if (azimuth != azimuth)
		return;

	if (innerDenuminator < 0.0)
		azimuth += PI;

	while (azimuth >= PI) azimuth -= 2.0 * PI;
	while (azimuth < -PI) azimuth += 2.0 * PI;

	double zenith = asin((cos(delta) * cos(tau) * cos(phi)) + (sin(delta) * sin(phi)));
	if (zenith != zenith)
		return;

	azimuth += m_AzimuthOffset;

	Zenith = static_cast<float> (zenith);
	Azimuth = static_cast<float> (azimuth);

	double cosZenith = cos(zenith);
	SunDirection.X = static_cast<float> (sin(azimuth) * cosZenith);
	SunDirection.Y = static_cast<float> (sin(zenith));
	SunDirection.Z = static_cast<float> (-cos(azimuth) * cosZenith);
	SunDirection.Normalize();
}

double USunPositionSimulatorComponent::normalizeAngleDeg(double angle)
{
	while (angle < 0) angle += 360.0;
	while (angle >= 360.0) angle -= 360.0;
	return angle;
}
