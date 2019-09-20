
#pragma once

#include "Simulation/Traffic/BehaviorTree/DeepDriveTBTDecoratorBase.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveTBTCheckOncomingTrafficDecorator, Log, All);

class DeepDriveTBTCheckOncomingTrafficDecorator	:	public	DeepDriveTBTDecoratorBase
{
public:

	DeepDriveTBTCheckOncomingTrafficDecorator(const FString &refLocName);

	virtual ~DeepDriveTBTCheckOncomingTrafficDecorator()	{	}

	virtual void bind(DeepDriveTrafficBlackboard &blackboard, DeepDrivePartialPath &path);

	virtual bool performCheck(DeepDriveTrafficBlackboard &blackboard, int32 pathPointIndex);

private:

	FString			m_ReferenceLocationName;

	int32			m_CheckClearanceIndex = -1;

	const float		CheckClearanceDistance = 1000.0f;

};
