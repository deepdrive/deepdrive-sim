
#pragma once

#include "Simulation/Traffic/BehaviorTree/DeepDriveTBTTaskBase.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveTBTCheckGreenToRedTask, Log, All);

class DeepDriveTBTCheckGreenToRedTask : public DeepDriveTBTTaskBase
{
public:

	DeepDriveTBTCheckGreenToRedTask(const FString &refLocationName, const FString &flagName);

	virtual ~DeepDriveTBTCheckGreenToRedTask()	{	}

	virtual void bind(DeepDriveTrafficBlackboard &blackboard, DeepDrivePartialPath &path);

	virtual bool execute(DeepDriveTrafficBlackboard &blackboard, float deltaSeconds, float &speed, int32 pathPointIndex);

private:

	FString			m_RefLocationName;
	FString			m_FlagName;

	int32			m_RefLocationIndex;

};
