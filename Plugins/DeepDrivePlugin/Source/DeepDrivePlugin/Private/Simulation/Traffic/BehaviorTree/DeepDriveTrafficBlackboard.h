
#pragma once

#include "CoreMinimal.h"

class DeepDrivePartialPath;
class ADeepDriveSimulation;
class ADeepDriveAgent;
struct SDeepDriveManeuver;

class DeepDriveTrafficBlackboard
{
public:

	DeepDriveTrafficBlackboard();

	void setBooleanValue(const FString &key, bool value);
	bool getBooleanValue(const FString &key, bool defaultValue = false) const;

	void setFloatValue(const FString &key, float value);
	float getFloatValue(const FString &key, float defaultValue = 0.0f) const;

	void setIntegerValue(const FString &key, int32 value);
	int32 getIntegerValue(const FString &key, int32 defaultValue = 0) const;

	void setVectorValue(const FString &key, const FVector &value);
	FVector getVectorValue(const FString &key, const FVector &defaultValue = FVector::ZeroVector) const;

	void setPartialPath(DeepDrivePartialPath &path);
	DeepDrivePartialPath* getPartialPath() const;

	void setManeuver(SDeepDriveManeuver &maneuver);
	SDeepDriveManeuver* getManeuver() const;

	void setAgent(ADeepDriveAgent &agent);
	ADeepDriveAgent* getAgent() const;

	void setSimulation(ADeepDriveSimulation &simulation);
	ADeepDriveSimulation* getSimulation() const;

private:

	TMap<FString, bool>				m_BooleanMap;
	TMap<FString, int32>			m_IntegerMap;
	TMap<FString, float>			m_FloatMap;
	TMap<FString, FVector>			m_VectorMap;

	DeepDrivePartialPath			*m_Path = 0;
	SDeepDriveManeuver				*m_Maneuver = 0;
	ADeepDriveAgent					*m_Agent = 0;
	ADeepDriveSimulation			*m_Simulation = 0;

};

inline void DeepDriveTrafficBlackboard::setPartialPath(DeepDrivePartialPath &path)
{
	m_Path = &path;
}

inline DeepDrivePartialPath *DeepDriveTrafficBlackboard::getPartialPath() const
{
	return m_Path;
}

inline void DeepDriveTrafficBlackboard::setManeuver(SDeepDriveManeuver &maneuver)
{
	m_Maneuver = &maneuver;
}

inline SDeepDriveManeuver *DeepDriveTrafficBlackboard::getManeuver() const
{
	return m_Maneuver;
}

inline void DeepDriveTrafficBlackboard::setAgent(ADeepDriveAgent &agent)
{
	m_Agent = &agent;
}

inline ADeepDriveAgent *DeepDriveTrafficBlackboard::getAgent() const
{
	return m_Agent;
}

inline void DeepDriveTrafficBlackboard::setSimulation(ADeepDriveSimulation &simulation)
{
	m_Simulation = &simulation;
}

inline ADeepDriveSimulation* DeepDriveTrafficBlackboard::getSimulation() const
{
	return m_Simulation;
}
