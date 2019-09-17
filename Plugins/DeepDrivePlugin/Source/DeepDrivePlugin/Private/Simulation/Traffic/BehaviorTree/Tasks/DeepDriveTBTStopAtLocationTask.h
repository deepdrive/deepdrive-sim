
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

	int32			m_StopLocationIndex = -1;
	int32			m_StopBeginIndex = -1;

	int32			m_SlowDownBeginIndex = -1;

	int32			m_IndexDelta = 0;

	bool			m_hasStopped = false;

	const float		StopBeginDistance = 400.0f;
	const float		SlowDownBeginDistance = 2000.0f;
};
