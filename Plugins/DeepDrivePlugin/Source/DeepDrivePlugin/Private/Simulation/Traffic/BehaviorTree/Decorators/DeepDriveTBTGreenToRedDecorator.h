
#pragma once

#include "Simulation/Traffic/BehaviorTree/DeepDriveTBTDecoratorBase.h"
#include "Simulation/TrafficLight/DeepDriveTrafficLight.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveTBTGreenToRedDecorator, Log, All);

class DeepDriveTBTGreenToRedDecorator	:	public	DeepDriveTBTDecoratorBase
{
public:

	virtual ~DeepDriveTBTGreenToRedDecorator()	{	}

	virtual void bind(DeepDriveTrafficBlackboard &blackboard, DeepDrivePartialPath &path);

	virtual bool performCheck(DeepDriveTrafficBlackboard &blackboard, int32 pathPointIndex);

private:

	int32				m_StopLineLocationIndex = -1;

};
