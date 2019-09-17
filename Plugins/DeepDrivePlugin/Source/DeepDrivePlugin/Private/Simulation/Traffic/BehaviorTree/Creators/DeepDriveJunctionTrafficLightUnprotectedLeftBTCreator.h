
#pragma once

#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBehaviorTreeCreatorBase.h"

class DeepDriveJunctionTrafficLightUnprotectedLeftBTCreator
{
public:

	static DeepDriveTrafficBehaviorTree *createBehaviorTree();

private:

	static bool registerCreator();

	static bool					s_isRegistered;

};
