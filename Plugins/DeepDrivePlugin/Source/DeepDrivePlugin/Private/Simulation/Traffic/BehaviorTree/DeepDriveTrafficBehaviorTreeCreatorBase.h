
#pragma once

#include "CoreMinimal.h"

class DeepDriveTrafficBehaviorTree;
struct SDeepDriveManeuver;

class DeepDriveTrafficBehaviorTreeCreatorBase
{
public:

	virtual ~DeepDriveTrafficBehaviorTreeCreatorBase()
		{	}

	virtual TArray<uint32> getKeys() = 0;

	virtual DeepDriveTrafficBehaviorTree *createBehaviorTree(SDeepDriveManeuver &maneuver) = 0;

};
