
#pragma once

#include "Simulation/Traffic/BehaviorTree/DeepDriveTBTDecoratorBase.h"
#include "Simulation/TrafficLight/DeepDriveTrafficLight.h"

class DeepDriveTBTIsInRangeDecorator	:	public	DeepDriveTBTDecoratorBase
{
public:

	DeepDriveTBTIsInRangeDecorator(const FString refLocationName, float range);

	virtual ~DeepDriveTBTIsInRangeDecorator()	{	}

	virtual void bind(DeepDriveTrafficBlackboard &blackboard, DeepDrivePartialPath &path);

	virtual bool performCheck(DeepDriveTrafficBlackboard &blackboard, int32 pathPointIndex);

private:

	FString						m_ReferenceLocationName;
	float						m_Range;

	int32						m_RangeBeginLocationIndex = -1;
	int32						m_RangeEndLocationIndex = -1;

};
