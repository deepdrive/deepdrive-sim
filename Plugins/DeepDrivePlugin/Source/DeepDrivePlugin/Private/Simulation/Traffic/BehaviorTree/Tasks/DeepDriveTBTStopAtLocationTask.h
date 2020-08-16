
#pragma once

#include "Simulation/Traffic/BehaviorTree/DeepDriveTBTTaskBase.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveTBTStopAtLocationTask, Log, All);

class DeepDriveTBTStopAtLocationTask : public DeepDriveTBTTaskBase
{
public:
	DeepDriveTBTStopAtLocationTask(const FString &stopLocationName, float exponent, float stopBeginDistance = 400.0f, float slowDownBeginDistance = 2000.0f);

	virtual ~DeepDriveTBTStopAtLocationTask()	{	}

	virtual void bind(DeepDriveTrafficBlackboard &blackboard, DeepDrivePartialPath &path);

	virtual bool execute(DeepDriveTrafficBlackboard &blackboard, float deltaSeconds, float &speed, int32 pathPointIndex);

private:

	FString			m_StopLocationName;
	float			m_Exponent = 1.0f;

	int32			m_StopLocationIndex = -1;
	int32			m_StopBeginIndex = -1;

	int32			m_SlowDownBeginIndex = -1;

	int32			m_IndexDelta = 0;

	const float		StopBeginDistance;
	const float		SlowDownBeginDistance;
};
