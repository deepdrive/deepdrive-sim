
#pragma once

#include "CoreMinimal.h"

class DeepDriveTrafficBlackboard;

class DeepDriveBehaviorTreeHelpers
{

public:

	static float calculateJunctionClearValue_new(DeepDriveTrafficBlackboard &blackboard, bool ignoreTrafficLights);

	static float calculateJunctionClearValue(DeepDriveTrafficBlackboard &blackboard, bool ignoreTrafficLights);
};
