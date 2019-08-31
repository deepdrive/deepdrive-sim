
#pragma once

#include "CoreMinimal.h"

class DeepDriveTrafficBlackboard;
class DeepDrivePartialPath;

class DeepDriveTBTTaskBase
{
public:

	virtual void bind(DeepDriveTrafficBlackboard &blackboard, DeepDrivePartialPath &path)
	{
	}

	virtual bool execute(DeepDriveTrafficBlackboard &blackboard, float deltaSeconds, float &speed, int32 pathPointIndex) = 0;

};
