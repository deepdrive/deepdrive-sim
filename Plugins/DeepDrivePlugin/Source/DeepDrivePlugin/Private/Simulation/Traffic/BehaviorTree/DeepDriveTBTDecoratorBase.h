
#pragma once

#include "CoreMinimal.h"

class DeepDriveTrafficBlackboard;

class DeepDriveTBTDecoratorBase
{
public:

	virtual bool performCheck(DeepDriveTrafficBlackboard &blackboard, int32 pathPointIndex) = 0;

};
