
#pragma once

#include "Simulation/Traffic/BehaviorTree/DeepDriveTBTTaskBase.h"

class DeepDriveTBTStopTask : public DeepDriveTBTTaskBase
{
public:

	virtual bool execute(DeepDriveTrafficBlackboard &blackboard, float deltaSeconds, float &speed, int32 pathPointIndex);

};
