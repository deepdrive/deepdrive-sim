#include "Simulation/Traffic/BehaviorTree/Tasks/DeepDriveTBTWaitTask.h"


DEFINE_LOG_CATEGORY(LogDeepDriveTBTWaitTask);

DeepDriveTBTWaitTask::DeepDriveTBTWaitTask(float waitTime)
	:	m_RemainingWaitTime(waitTime)
{

}

bool DeepDriveTBTWaitTask::execute(DeepDriveTrafficBlackboard &blackboard, float deltaSeconds, float &speed, int32 pathPointIndex)
{
	if (m_isExpired == false)
	{
		if (m_RemainingWaitTime > 0.0f)
		{
			m_RemainingWaitTime -= deltaSeconds;
			speed = 0.0f;
		}
		m_isExpired = m_RemainingWaitTime <= 0.0f;
		if(m_isExpired)
			UE_LOG(LogDeepDriveTBTWaitTask, Log, TEXT("DeepDriveTBTWaitTask isExpired") );
	}
	return !m_isExpired;
}
