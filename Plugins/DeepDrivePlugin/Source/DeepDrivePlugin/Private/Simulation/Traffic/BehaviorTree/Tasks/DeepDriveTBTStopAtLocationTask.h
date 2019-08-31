
#pragma once

#include "Simulation/Traffic/BehaviorTree/DeepDriveTBTTaskBase.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveTBTStopAtLocationTask, Log, All);

class DeepDriveTBTStopAtLocationTask : public DeepDriveTBTTaskBase
{
public:

	DeepDriveTBTStopAtLocationTask();

	virtual void bind(DeepDriveTrafficBlackboard &blackboard, DeepDrivePartialPath &path);

	virtual bool execute(DeepDriveTrafficBlackboard &blackboard, float deltaSeconds, float &speed, int32 pathPointIndex);

private:

	int32			m_StopLineLocationIndex = -1;

	int32			m_SlowDownBeginIndex = -1;
	int32			m_StopRangeBeginIndex = -1;

	int32			m_IndexDelta = 0;

	float			m_SlowDownBeginDistance = 0.0f;

	bool			m_hasStopped = false;


	const float		StopRangeBeginDistance = 300.0f;
	const float		SlowDownBeginDistance = 2000.0f;
};
