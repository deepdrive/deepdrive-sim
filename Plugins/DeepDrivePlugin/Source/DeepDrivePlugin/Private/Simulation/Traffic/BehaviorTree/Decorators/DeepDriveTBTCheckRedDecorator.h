
#pragma once

#include "Simulation/Traffic/BehaviorTree/DeepDriveTBTDecoratorBase.h"
#include "Simulation/TrafficLight/DeepDriveTrafficLight.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveTBTCheckRedDecorator, Log, All);

class DeepDriveTBTCheckRedDecorator	:	public	DeepDriveTBTDecoratorBase
{
public:

	virtual bool performCheck(DeepDriveTrafficBlackboard &blackboard, int32 pathPointIndex);

private:

};
