// Fill out your copyright notice in the Description page of Project Settings.

#include "DeepDrivePluginPrivatePCH.h"

#include "Public/Simulation/Environment/SunLightSimulatorComponent.h"


UTexture2D* USunLightSimulatorComponent::CreateLookUpTexture(int32 Width, int32 NumSamples)
{
	UTexture2D *texture = 0;

	texture = UTexture2D::CreateTransient(Width, 2, EPixelFormat::PF_R32_FLOAT);
	if(texture)
	{
		setupTable(Width, NumSamples);

		FTexture2DMipMap& Mip = texture->PlatformData->Mips[0];
		void* data = Mip.BulkData.Lock( LOCK_READ_WRITE );
		const int32 rowSize = Width * sizeof(float);
		FMemory::Memcpy( data, m_RayleighTable.GetData(), rowSize );
		FMemory::Memcpy( reinterpret_cast<uint8*> (data) + rowSize, m_MieTable.GetData(), rowSize );
		Mip.BulkData.Unlock( );
		texture->UpdateResource();
	}

	return texture;
}


void USunLightSimulatorComponent::setupTable(int32 width, int32 numSamples)
{
	const float dAng = PI / static_cast<float> (width);
	float  curAng = 0.0f;
	for(int32 i = 0; i < width; ++i)
	{
		FVector dir(FGenericPlatformMath::Sin(curAng), FGenericPlatformMath::Cos(curAng), 0.0f);

		float odR = 0.0f;
		float odM = 0.0f;
		calculateOpticalDepth(1.59f, dir, numSamples, odR, odM);

		m_RayleighTable.Add(odR);
		m_MieTable.Add(odM);

		curAng += dAng;
	}
}

void USunLightSimulatorComponent::calculateOpticalDepth(float height, const FVector &dir, int32 numSamples, float &odRayleigh, float &odMie)
{
	FVector start(0.0f, height + m_EarthRadius, 0.0f);

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
	const float t1 = (-b - FMath::Sqrt(d)) / (2.0f * a);
	return t0;
}
