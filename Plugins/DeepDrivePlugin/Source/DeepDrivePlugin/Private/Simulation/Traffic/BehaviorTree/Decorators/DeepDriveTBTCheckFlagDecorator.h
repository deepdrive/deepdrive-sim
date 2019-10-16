
#pragma once

#include "Simulation/Traffic/BehaviorTree/DeepDriveTBTDecoratorBase.h"
#include "Simulation/TrafficLight/DeepDriveTrafficLight.h"

class DeepDriveTBTCheckFlagDecorator	:	public	DeepDriveTBTDecoratorBase
{
public:

	DeepDriveTBTCheckFlagDecorator(const FString &flagName, bool refValue, bool defaultValue);

	virtual ~DeepDriveTBTCheckFlagDecorator()	{	}

	virtual bool performCheck(DeepDriveTrafficBlackboard &blackboard, int32 pathPointIndex) const;

private:

	FString				m_FlagName;
	bool				m_RefValue;
	bool				m_DefaultValue;
};
