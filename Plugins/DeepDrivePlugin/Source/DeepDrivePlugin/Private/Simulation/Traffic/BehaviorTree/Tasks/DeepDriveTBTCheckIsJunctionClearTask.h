
#pragma once

#include "Simulation/Traffic/BehaviorTree/DeepDriveTBTTaskBase.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveTBTCheckIsJunctionClearTask, Log, All);

class DeepDriveTBTCheckIsJunctionClearTask : public DeepDriveTBTTaskBase
{
public:
	DeepDriveTBTCheckIsJunctionClearTask(const FString &refLocationName, float distance, bool ignoreTrafficLights);

	DeepDriveTBTCheckIsJunctionClearTask(const FString &refLocationName, float distance, bool ignoreTrafficLights, const FString &flagName);

	virtual void bind(DeepDriveTrafficBlackboard &blackboard, DeepDrivePartialPath &path);

	virtual bool execute(DeepDriveTrafficBlackboard &blackboard, float deltaSeconds, float &speed, int32 pathPointIndex);

private:

	FString			m_RefLocationName;
	float			m_Distance;
	bool			m_ignoreTrafficLights;
	FString			m_FlagName;

	int32			m_RefLocationIndex;

};
