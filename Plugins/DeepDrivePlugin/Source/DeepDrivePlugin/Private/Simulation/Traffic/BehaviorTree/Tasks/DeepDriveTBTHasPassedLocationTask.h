
#pragma once

#include "Simulation/Traffic/BehaviorTree/DeepDriveTBTTaskBase.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveTBTHasPassedLocationTask, Log, All);

class DeepDriveTBTHasPassedLocationTask : public DeepDriveTBTTaskBase
{
public:

	DeepDriveTBTHasPassedLocationTask(const FString &locationName, float distance, const FString &flagName);

	virtual ~DeepDriveTBTHasPassedLocationTask()	{	}

	virtual void bind(DeepDriveTrafficBlackboard &blackboard, DeepDrivePartialPath &path);

	virtual bool execute(DeepDriveTrafficBlackboard &blackboard, float deltaSeconds, float &speed, int32 pathPointIndex);

private:

	FString			m_LocationName;
	float			m_Distance;
	FString			m_FlagName;

	int32			m_LocationIndex = -1;
};
