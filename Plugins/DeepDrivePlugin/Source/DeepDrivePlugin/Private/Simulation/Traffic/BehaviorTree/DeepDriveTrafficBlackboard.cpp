
#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBlackboard.h"

DeepDriveTrafficBlackboard::DeepDriveTrafficBlackboard()
	:	m_BooleanMap()
	,	m_IntegerMap()
	,	m_FloatMap()
	,	m_VectorMap()
{
}

void DeepDriveTrafficBlackboard::setBooleanValue(const FString &key, bool value)
{
	if(m_BooleanMap.Contains(key))
		m_BooleanMap[key] = value;
	else
		m_BooleanMap.Add(key, value);
}

bool DeepDriveTrafficBlackboard::getBooleanValue(const FString &key, bool defaultValue)
{
	return m_BooleanMap.Contains(key) ? m_BooleanMap[key] : defaultValue;
}

void DeepDriveTrafficBlackboard::setFloatValue(const FString &key, float value)
{
	if(m_FloatMap.Contains(key))
		m_FloatMap[key] = value;
	else
		m_FloatMap.Add(key, value);
}

float DeepDriveTrafficBlackboard::getFloatValue(const FString &key, float defaultValue)
{
	return m_FloatMap.Contains(key) ? m_FloatMap[key] : defaultValue;
}

void DeepDriveTrafficBlackboard::setIntegerValue(const FString &key, int32 value)
{
	if(m_IntegerMap.Contains(key))
		m_IntegerMap[key] = value;
	else
		m_IntegerMap.Add(key, value);
}

int32 DeepDriveTrafficBlackboard::getIntegerValue(const FString &key, int32 defaultValue)
{
	return m_IntegerMap.Contains(key) ? m_IntegerMap[key] : defaultValue;
}

void DeepDriveTrafficBlackboard::setVectorValue(const FString &key, const FVector &value)
{
	if(m_VectorMap.Contains(key))
		m_VectorMap[key] = value;
	else
		m_VectorMap.Add(key, value);
}

FVector DeepDriveTrafficBlackboard::getVectorValue(const FString &key, const FVector &defaultValue)
{
	return m_VectorMap.Contains(key) ? m_VectorMap[key] : defaultValue;
}
