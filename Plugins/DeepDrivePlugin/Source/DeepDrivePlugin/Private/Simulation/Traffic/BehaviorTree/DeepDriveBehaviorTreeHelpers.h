
#pragma once

#include "CoreMinimal.h"

class DeepDriveTrafficBlackboard;

class DeepDriveBehaviorTreeHelpers
{

public:

	static bool isJunctionClear(DeepDriveTrafficBlackboard &blackboard);

	static float getJunctionClearance(DeepDriveTrafficBlackboard &blackboard);
};
