

#include "DeepDrivePluginPrivatePCH.h"
#include "DeepDriveRandomStream.h"


void UDeepDriveRandomStream::initialize(int32 seed)
{
	m_RandomStream = FRandomStream(seed);
}

int32 UDeepDriveRandomStream::RandomInteger(int32 Max)
{
	return m_RandomStream.RandHelper(Max);
}

int32 UDeepDriveRandomStream::RandomIntegerInRange(int32 Min, int32 Max)
{
	return m_RandomStream.RandRange(Min, Max);
}