// Fill out your copyright notice in the Description page of Project Settings.

#include "DeepDrivePluginPrivatePCH.h"

#include "Public/Simulation/Environment/SunSimulationComponent.h"

#undef UpdateResource


DEFINE_LOG_CATEGORY(LogSunSimulationComponent);

void USunSimulationComponent::Advance(float DeltaSeconds)
{

	switch(m_SimulationMode)
	{
		case Idle:
			break;

		case Forward:
			advanceTime(DeltaSeconds);
			if(0 /* desired time reached */)
				m_SimulationMode = Idle;
			break;

		case Continous:
			advanceTime(DeltaSeconds);
			break;
	}

	calculatePosition();

}


UTexture2D* USunSimulationComponent::CreateLookUpTexture(int32 Width, int32 NumSamples)
{
	UTexture2D *texture = 0;

	texture = UTexture2D::CreateTransient(Width, 4, EPixelFormat::PF_R8G8B8A8);
	if(texture)
	{
		setupTable(Width, NumSamples);

		FTexture2DMipMap& Mip = texture->PlatformData->Mips[0];
		void* data = Mip.BulkData.Lock( LOCK_READ_WRITE );
		const int32 stride = 4 * Width;

		uint8 *ptr = reinterpret_cast<uint8*> (data);

		for(signed i = 0; i < Width; ++i)
		{
			uint32 rayleigh = static_cast<uint32> (m_RayleighTable[i]);
			ptr[stride + 0] = ptr[0] = (rayleigh >> 16) & 0xFF;
			ptr[stride + 1] = ptr[1] = (rayleigh >> 8) & 0xFF;
			ptr[stride + 2] = ptr[2] = rayleigh & 0xFF;

			uint32 mie = static_cast<uint32> (m_MieTable[i]);
			ptr[3 * stride + 0] = ptr[2 * stride + 0] = (mie >> 16) & 0xFF;
			ptr[3 * stride + 1] = ptr[2 * stride + 1] = (mie >> 8) & 0xFF;
			ptr[3 * stride + 2] = ptr[2 * stride + 2] = mie & 0xFF;

			ptr += 4;
		}

		Mip.BulkData.Unlock( );
		texture->UpdateResource();
	}

	return texture;
}

FLinearColor USunSimulationComponent::CalculateSunLightColor(const FVector &SunDirection)
{
	FVector sunLightColor;

	int32 numSamples = 10;
	FVector samplePos = FVector(0.0f, 0.0f, m_EarthRadius + m_RefHeight);
	float t = intersect(samplePos, SunDirection);

	FVector deltaPos = SunDirection * t / static_cast<float> (numSamples);
	float segmentLength = t / numSamples;

	samplePos += SunDirection * 0.5f * segmentLength;

	FVector sumR(0.0f, 0.0f, 0.0f);
	FVector sumM(0.0f, 0.0f, 0.0f);
	float opticalDepthR = 0;
	float opticalDepthM = 0;

	for (signed i = 0; i < numSamples; ++i)
	{
		float height = samplePos.Size() - m_EarthRadius;
		float hr = exp(-height / m_Hr) * segmentLength * 1000.0f;
		float hm = exp(-height / m_Hm) * segmentLength * 1000.0f;
		opticalDepthR += hr;
		opticalDepthM += hm;

		height = FMath::Clamp(height, 0.0f, m_AtmosphereRadius - m_EarthRadius);
		float opticalDepthLightR = 0.0f;
		float opticalDepthLightM = 0.0f;
		calculateOpticalDepth(height, SunDirection, numSamples, opticalDepthLightR, opticalDepthLightM);

		if(opticalDepthLightR > 0.0 && opticalDepthLightM > 0.0)
		{
			FVector tau	(	m_BetaR[0] * (opticalDepthR + opticalDepthLightR) + m_BetaM[0] * 1.1f * (opticalDepthM + opticalDepthLightM)
						,	m_BetaR[1] * (opticalDepthR + opticalDepthLightR) + m_BetaM[1] * 1.1f * (opticalDepthM + opticalDepthLightM)
						,	m_BetaR[2] * (opticalDepthR + opticalDepthLightR) + m_BetaM[2] * 1.1f * (opticalDepthM + opticalDepthLightM)
						);
			FVector attenuation = FVector(exp(-tau[0]), exp(-tau[1]), exp(-tau[2]));
			sumR += attenuation * hr;
			sumM += attenuation * hm;
		}

		samplePos += deltaPos;
	}

	float g = 0.98f;
	float phaseR = 3.0f / (16.0f * PI) * 2.0f;
	float phaseM = 3.0f / (8.0f * PI) * ((1 - g * g) * 2.0f)/((2.0f + g * g) * pow(1.0f + g * g - 2 * g, 1.5f));
	sunLightColor.X = 20.0f * (sumR[0] * phaseR * m_BetaR[0] + sumM[0] * phaseM * m_BetaM[0]);
	sunLightColor.Y = 20.0f * (sumR[1] * phaseR * m_BetaR[1] + sumM[1] * phaseM * m_BetaM[1]);
	sunLightColor.Z = 20.0f * (sumR[2] * phaseR * m_BetaR[2] + sumM[2] * phaseM * m_BetaM[2]);

	float maxVal = FMath::Max3(sunLightColor[0], sunLightColor[1], sunLightColor[2]);
	if(maxVal > 0.0001)
		sunLightColor *= 1.0 / maxVal;

	return FLinearColor(FMath::Max(sunLightColor.X, 0.2f), FMath::Max(sunLightColor.Y, 0.0f), FMath::Max(sunLightColor.Z, 0.2f), 1.0);
}


void USunSimulationComponent::SetLocation(float LongitudeDegrees, float LatitudeDegrees)
{
	m_LongitudeDegrees = LongitudeDegrees;
	m_LatitudeDegrees = LatitudeDegrees;
}

void USunSimulationComponent::SetDate(int32 Year, int32 Month, int32 Day, float Duration)
{
	m_desiredDateTime = FDateTime(Year, Month, Day, m_desiredDateTime.GetHour(), m_desiredDateTime.GetMinute());
	prepareTimespan(Duration);
}

void USunSimulationComponent::SetTime(int32 Hour, int32 Minute, float Duration)
{
	m_desiredDateTime = FDateTime(m_desiredDateTime.GetYear(), m_desiredDateTime.GetMonth(), m_desiredDateTime.GetDay(), Hour, Minute);
	prepareTimespan(Duration);
}

void USunSimulationComponent::SetDateAndTime(int32 Year, int32 Month, int32 Day, int32 Hour, int32 Minute, float Duration)
{
	m_desiredDateTime = FDateTime(Year, Month, Day, Hour, Minute);
	prepareTimespan(Duration);
}


void USunSimulationComponent::prepareTimespan(float duration)
{
	if (duration <= 0.0f)
	{
		m_curDateTime = m_desiredDateTime;
		m_curTimeSpan = FTimespan::Zero();
		m_SimulationMode = Idle;
	}
	else
	{
		m_curTimeSpan = m_desiredDateTime - m_curDateTime;
		m_curTimeSpan /= duration;
		m_SimulationMode = Forward;
	}
}


void USunSimulationComponent::EnableSimulation(bool Enabled)
{
	if(Enabled)
	{
		if(m_SimulationMode != Continous)
		{
			int32 seconds = static_cast<float> (m_MinutesPerSecond * 60.0f);
			m_curTimeSpan = FTimespan(0, 0, seconds);
			m_SimulationMode = Continous;
		}
	}
	else
	{
		m_SimulationMode = Idle;
	}
}

void USunSimulationComponent::SetSimulationSpeed(int32 MinutesPerSecond)
{
	m_MinutesPerSecond = MinutesPerSecond;
	if (m_SimulationMode == Continous)
	{
		int32 seconds = static_cast<float> (m_MinutesPerSecond * 60.0f);
		m_curTimeSpan = FTimespan(0, 0, seconds);
	}
}

void USunSimulationComponent::setupTable(int32 width, int32 numSamples)
{
	const float dAng = PI / static_cast<float> (width);
	float  curAng = 0.0f;
	for(int32 i = 0; i < width; ++i)
	{
		FVector dir(0.0f, FGenericPlatformMath::Cos(curAng), FGenericPlatformMath::Sin(curAng));

		float odR = 0.0f;
		float odM = 0.0f;
		calculateOpticalDepth(m_RefHeight, dir, numSamples, odR, odM);

		// UE_LOG(LogSunSimulationComponent, Log, TEXT("%f %f %f"), curAng, odR, odM);

		m_RayleighTable.Add(odR);
		m_MieTable.Add(odM);

		curAng += dAng;
	}
}

void USunSimulationComponent::calculateOpticalDepth(float height, const FVector &dir, int32 numSamples, float &odRayleigh, float &odMie)
{
	FVector start(0.0f, 0.0f, height + m_EarthRadius);

	odRayleigh = 0.0f;
	odMie = 0.0f;

	float tRay = intersect(start, dir);

	float segmentLength = tRay / static_cast<float> (numSamples);
	float tCurrent = 0.0f;
	for (int32 j = 0; j < numSamples; ++j)
	{
		FVector samplePosition = start + dir * (tCurrent + 0.5 * segmentLength);

		const float curSampleHeight = samplePosition.Size() - m_EarthRadius;
		if (curSampleHeight < 0)
		{
			odRayleigh = 0;
			odMie = 0;
			break;
		}

		odRayleigh += FGenericPlatformMath::Exp(-curSampleHeight / m_Hr);
		odMie += FGenericPlatformMath::Exp(-curSampleHeight / m_Hm);
		tCurrent += segmentLength;
	}

	odRayleigh *= 1000.0 * segmentLength;
	odMie *= 1000.0 * segmentLength;
}

float USunSimulationComponent::intersect(const FVector &start, const FVector &dir)
{
	const float a = FVector::DotProduct(dir, dir);
	const float b = 2.0f * FVector::DotProduct(dir, start);
	const float c = FVector::DotProduct(start, start) - m_AtmosphereRadius * m_AtmosphereRadius;

	const float d = b * b - 4.0f * a * c;

	const float t0 = (-b + FMath::Sqrt(d)) / (2.0f * a);
	return t0;
}




double USunSimulationComponent::getJulianDay()
{
	// see http://en.wikipedia.org/wiki/Julian_day#Converting_Julian_or_Gregorian_calendar_date_to_Julian_Day_Number

	double m = (m_curDateTime.GetMonth() - 14.0) / 12.0;

	double a = 1461.0 * (m_curDateTime.GetYear() + 4800.0 + m) / 4.0;
	double b = (367.0 * (m_curDateTime.GetMonth() - 2.0 - 12.0 * (m))) / 12.0;
	
	double julianDay = a + b - (3.0 * ((m_curDateTime.GetYear() + 4900.0 + m) / 100.0)) / 4.0 + m_curDateTime.GetDay() - 32075.0;
	return julianDay;
}

double USunSimulationComponent::getJulianDate(double julianDay)
{
	double julianDate = julianDay;
	julianDate += (m_curDateTime.GetHour() - 12.0) / 24.0;
	julianDate += (m_curDateTime.GetMinute()) / 1440.0;
	return julianDate;
}

void USunSimulationComponent::advanceTime(float DeltaSeconds)
{
	m_curDateTime = m_curDateTime + m_curTimeSpan * DeltaSeconds;
	if (m_SimulationMode == Forward)
	{
		if (m_curTimeSpan < FTimespan::Zero())
		{
			if (m_curDateTime <= m_desiredDateTime)
			{
				m_curDateTime = m_desiredDateTime;
				m_SimulationMode = Idle;
			}
		}
		else if (m_curTimeSpan > FTimespan::Zero())
		{
			if (m_curDateTime >= m_desiredDateTime)
			{
				m_curDateTime = m_desiredDateTime;
				m_SimulationMode = Idle;
			}
		}
	}
	m_JulianDay = getJulianDay();
}


void USunSimulationComponent::calculatePosition()
{
	const double dayTimeDecimal = m_curDateTime.GetHour() + (m_curDateTime.GetMinute() + m_curDateTime.GetSecond() / 60.0) / 60.0;
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
	double localMeanSiderealTime = deg2rad(greenwichMeanSiderealTime * 15.0 + m_LongitudeDegrees);
	double latRadians = deg2rad(m_LatitudeDegrees);
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

double USunSimulationComponent::normalizeAngleDeg(double angle)
{
	while (angle < 0) angle += 360.0;
	while (angle >= 360.0) angle -= 360.0;
	return angle;
}
