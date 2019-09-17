
#pragma once

#include "Simulation/Traffic/BehaviorTree/DeepDriveTBTTaskBase.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveTBTWaitForClearance, Log, All);

class DeepDriveTBTWaitForClearance : public DeepDriveTBTTaskBase
{
public:

	DeepDriveTBTWaitForClearance();
	
	virtual bool execute(DeepDriveTrafficBlackboard &blackboard, float deltaSeconds, float &speed, int32 pathPointIndex);

private:

};
