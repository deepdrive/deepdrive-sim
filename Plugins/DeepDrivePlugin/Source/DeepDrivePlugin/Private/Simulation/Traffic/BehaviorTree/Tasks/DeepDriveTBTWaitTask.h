
#pragma once

#include "Simulation/Traffic/BehaviorTree/DeepDriveTBTTaskBase.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveTBTWaitTask, Log, All);

class DeepDriveTBTWaitTask : public DeepDriveTBTTaskBase
{
public:

	DeepDriveTBTWaitTask(float waitTime);
	
	virtual bool execute(DeepDriveTrafficBlackboard &blackboard, float deltaSeconds, float &speed, int32 pathPointIndex);

private:

	float			m_RemainingWaitTime = 0.0f;
	bool			m_isExpired = false;

};
