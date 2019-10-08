
#pragma once

#include "Simulation/Traffic/BehaviorTree/DeepDriveTBTDecoratorBase.h"
#include "Simulation/TrafficLight/DeepDriveTrafficLight.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveTBTIsJunctionClearDecorator, Log, All);

class DeepDriveTBTIsJunctionClearDecorator	:	public	DeepDriveTBTDecoratorBase
{
public:

	DeepDriveTBTIsJunctionClearDecorator(bool checkOnce);

	virtual ~DeepDriveTBTIsJunctionClearDecorator()	{	}

	virtual bool performCheck(DeepDriveTrafficBlackboard &blackboard, int32 pathPointIndex);

private:

	bool			m_CheckOnce;

	bool			m_isJunctionClear = false;
};
