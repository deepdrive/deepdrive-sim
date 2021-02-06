// Fill out your copyright notice in the Description page of Project Settings.


#include "Simulation/Environment/SunPositionSimulatorComponent.h"


void USunPositionSimulatorComponent::SetDate(int32 year, int32 month, int32 day, int32 hour)
{
	month = FMath::Clamp<uint16>(month, 1, 12);
	day = FMath::Clamp<uint16>(day, 1, 31);
	hour = FMath::Clamp<uint16>(hour, 0, 23);

	m_DateTime = FDateTime(year, month, day, hour, 0, 0, 0);

	m_JulianDay = getJulianDay();
}


double USunPositionSimulatorComponent::getJulianDay()
{
	// see http://en.wikipedia.org/wiki/Julian_day#Converting_Julian_or_Gregorian_calendar_date_to_Julian_Day_Number

	double m = (m_DateTime.GetMonth() - 14.0) / 12.0;

	double a = 1461.0 * (m_DateTime.GetYear() + 4800.0 + m) / 4.0;
	double b = (367.0 * (m_DateTime.GetMonth() - 2.0 - 12.0 * (m))) / 12.0;
	
	double julianDay = a + b - (3.0 * ((m_DateTime.GetYear() + 4900.0 + m) / 100.0)) / 4.0 + m_DateTime.GetDay() - 32075.0;
	return julianDay;
}

double USunPositionSimulatorComponent::getJulianDate(double julianDay)
{
	double julianDate = julianDay;
	julianDate += (m_DateTime.GetHour() - 12.0) / 24.0;
	julianDate += (m_DateTime.GetMinute()) / 1440.0;
	return julianDate;
}

void USunPositionSimulatorComponent::forward(float minutes)
{
	int32 seconds = static_cast<float> (minutes * 60.0f);
	FTimespan timespan(0, 0, seconds);
	m_DateTime = m_DateTime + timespan;
	m_JulianDay = getJulianDay();
}

void USunPositionSimulatorComponent::Advance(float DeltaSeconds)
{
	forward(MinutesPerSecond * DeltaSeconds);
	advancePSA(DeltaSeconds);
}

void USunPositionSimulatorComponent::advancePSA(float DeltaSeonds)
{
	const double dayTimeDecimal = m_DateTime.GetHour() + (m_DateTime.GetMinute() + m_DateTime.GetSecond() / 60.0) / 60.0;
	double julianDate = getJulianDay() - 0.5 + dayTimeDecimal / 24.0;
	double elapsedJulianDays = julianDate - 2451545.0;

	const double meanLongitude = 4.8950630 + 0.017202791698 * elapsedJulianDays;
	const double meanAnomaly  = 6.2400600 + 0.0172019699 * elapsedJulianDays;
	const double omega = 2.1429 - 0.0010394594 * elapsedJulianDays;
	const double eclipticLongitude = meanLongitude + 0.03341607 * sin(meanAnomaly) + 0.00034894 * sin(2.0 * meanAnomaly) - 0.0001134 - 0.0000203 * sin(omega);
	const double eclipticObliquity = 0.4090928 - 6.2140e-9 * elapsedJulianDays + 0.0000396 * cos(omega);

	const double sinEclipticLongitude = sin(eclipticLongitude);
	double dY = cos(eclipticObliquity) * sinEclipticLongitude;
	double dX = cos(eclipticLongitude);
	double rightAscension = atan2(dY, dX);
	if (rightAscension < 0.0)
		rightAscension = rightAscension + 2.0 * PI;
	double declination = asin(sin(eclipticObliquity) * sinEclipticLongitude);

	double greenwichMeanSiderealTime = 6.6974243242 + 0.0657098283 * elapsedJulianDays	+ dayTimeDecimal;
	double localMeanSiderealTime = deg2rad(greenwichMeanSiderealTime * 15.0 + LongitudeDegrees);
	double latRadians = deg2rad(LatitudeDegrees);
	double hourAngle = localMeanSiderealTime - rightAscension;
	double cosLatitude = cos(latRadians);
	double sinLatitude = sin(latRadians);
	double cosHourAngle = cos(hourAngle);

	double zenith = static_cast<float> (acos(cosLatitude * cosHourAngle * cos(declination) + sin(declination) * sinLatitude));

	dY = -sin(hourAngle);
	dX = tan(declination) * cosLatitude - sinLatitude * cosHourAngle;

	double azi = static_cast<float> (atan2(dY, dX));
	if (azi < 0.0)
		azi += 2.0 * PI;
	Azimuth = static_cast<float>(rad2deg(azi));
	// Parallax Correction
	double parallax = (6371.01 / 149597890.0) * sin(zenith);
	Zenith = static_cast<float> (rad2deg(zenith + parallax));
}



void USunPositionSimulatorComponent::advanceClassic(float DeltaSeonds)
{
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

	double tau = deg2rad((meanStarTime * 15) + LongitudeDegrees - rightAscension);
	double phi = deg2rad(LatitudeDegrees);
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
