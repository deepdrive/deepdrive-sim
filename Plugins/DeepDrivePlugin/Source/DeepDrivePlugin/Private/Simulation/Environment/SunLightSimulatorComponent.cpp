// Fill out your copyright notice in the Description page of Project Settings.

#include "DeepDrivePluginPrivatePCH.h"

#include "Public/Simulation/Environment/SunLightSimulatorComponent.h"

#undef UpdateResource


DEFINE_LOG_CATEGORY(LogSunLightSimulator);

UTexture2D* USunLightSimulatorComponent::CreateLookUpTexture(int32 Width, int32 NumSamples)
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

FVector USunLightSimulatorComponent::CalculateSunLightColor(const FVector &SunDirection)
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

	return sunLightColor;
}


void USunLightSimulatorComponent::setupTable(int32 width, int32 numSamples)
{
	const float dAng = PI / static_cast<float> (width);
	float  curAng = 0.0f;
	for(int32 i = 0; i < width; ++i)
	{
		FVector dir(0.0f, FGenericPlatformMath::Cos(curAng), FGenericPlatformMath::Sin(curAng));

		float odR = 0.0f;
		float odM = 0.0f;
		calculateOpticalDepth(m_RefHeight, dir, numSamples, odR, odM);

		UE_LOG(LogSunLightSimulator, Log, TEXT("%f %f %f"), curAng, odR, odM);

		m_RayleighTable.Add(odR);
		m_MieTable.Add(odM);

		curAng += dAng;
	}
}

void USunLightSimulatorComponent::calculateOpticalDepth(float height, const FVector &dir, int32 numSamples, float &odRayleigh, float &odMie)
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

float USunLightSimulatorComponent::intersect(const FVector &start, const FVector &dir)
{
	const float a = FVector::DotProduct(dir, dir);
	const float b = 2.0f * FVector::DotProduct(dir, start);
	const float c = FVector::DotProduct(start, start) - m_AtmosphereRadius * m_AtmosphereRadius;

	const float d = b * b - 4.0f * a * c;

	const float t0 = (-b + FMath::Sqrt(d)) / (2.0f * a);
	return t0;
}
