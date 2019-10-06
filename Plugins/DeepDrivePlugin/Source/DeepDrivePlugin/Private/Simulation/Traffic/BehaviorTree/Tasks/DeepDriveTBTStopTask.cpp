#include "Simulation/Traffic/BehaviorTree/Tasks/DeepDriveTBTStopTask.h"


bool DeepDriveTBTStopTask::execute(DeepDriveTrafficBlackboard &blackboard, float deltaSeconds, float &speed, int32 pathPointIndex)
{
	speed = 0.0f;
	return true;
}
