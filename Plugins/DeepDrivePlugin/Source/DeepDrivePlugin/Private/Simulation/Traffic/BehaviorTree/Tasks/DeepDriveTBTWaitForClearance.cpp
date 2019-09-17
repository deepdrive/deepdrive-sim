
#include "Simulation/Traffic/BehaviorTree/Tasks/DeepDriveTBTWaitForClearance.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveBehaviorTreeHelpers.h"

DEFINE_LOG_CATEGORY(LogDeepDriveTBTWaitForClearance);

DeepDriveTBTWaitForClearance::DeepDriveTBTWaitForClearance()
{
}

bool DeepDriveTBTWaitForClearance::execute(DeepDriveTrafficBlackboard &blackboard, float deltaSeconds, float &speed, int32 pathPointIndex)
{
	bool isJunctionClear = DeepDriveBehaviorTreeHelpers::isJunctionClear(blackboard);

	if(isJunctionClear == false)
		speed = 0.0f;

	return !isJunctionClear;
}
