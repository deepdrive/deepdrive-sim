
#pragma once

#include "CoreMinimal.h"

class DeepDriveTrafficBlackboard;

class DeepDriveBehaviorTreeHelpers
{

public:

	static float calculateJunctionClearValue(DeepDriveTrafficBlackboard &blackboard, bool ignoreTrafficLights);
};
