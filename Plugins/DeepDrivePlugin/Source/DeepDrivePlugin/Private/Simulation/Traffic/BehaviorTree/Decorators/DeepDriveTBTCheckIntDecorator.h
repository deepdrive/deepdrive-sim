
#pragma once

#include "Simulation/Traffic/BehaviorTree/DeepDriveTBTDecoratorBase.h"
#include "Simulation/TrafficLight/DeepDriveTrafficLight.h"

class DeepDriveTBTCheckIntDecorator	:	public	DeepDriveTBTDecoratorBase
{
public:

	DeepDriveTBTCheckIntDecorator(const FString &flagName, int32 refValue, int32 defaultValue);

	virtual ~DeepDriveTBTCheckIntDecorator()	{	}

	virtual bool performCheck(DeepDriveTrafficBlackboard &blackboard, int32 pathPointIndex) const;

private:

	FString				m_FlagName;
	int32				m_RefValue;
	int32				m_DefaultValue;
};
