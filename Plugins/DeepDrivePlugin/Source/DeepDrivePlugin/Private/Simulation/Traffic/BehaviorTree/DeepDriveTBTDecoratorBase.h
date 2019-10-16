
#pragma once

#include "CoreMinimal.h"

class DeepDriveTrafficBlackboard;
class DeepDrivePartialPath;

class DeepDriveTBTDecoratorBase
{
public:

	virtual ~DeepDriveTBTDecoratorBase()
	{
	}

	virtual void bind(DeepDriveTrafficBlackboard &blackboard, DeepDrivePartialPath &path)
	{
	}

	virtual bool performCheck(DeepDriveTrafficBlackboard &blackboard, int32 pathPointIndex) const = 0;
};
