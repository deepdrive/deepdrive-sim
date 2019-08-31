
#include "Simulation/Traffic/BehaviorTree/Tasks/DeepDriveTBTStopAtLocationTask.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBlackboard.h"
#include "Simulation/Agent/DeepDriveAgent.h"

#include "Private/Simulation/Traffic/Path/DeepDrivePartialPath.h"

DEFINE_LOG_CATEGORY(LogDeepDriveTBTStopAtLocationTask);

DeepDriveTBTStopAtLocationTask::DeepDriveTBTStopAtLocationTask()
{
}

void DeepDriveTBTStopAtLocationTask::bind(DeepDriveTrafficBlackboard &blackboard, DeepDrivePartialPath &path)
{
	FVector stopLineLocation = blackboard.getVectorValue("StopLineLocation", FVector::ZeroVector);

	m_StopLineLocationIndex = path.findClosestPathPoint(stopLineLocation, 0);
	if(m_StopLineLocationIndex >= 0)
	{
		const float frontBumperDist = blackboard.getAgent()->getFrontBumperDistance();
		m_StopRangeBeginIndex = path.rewind(m_StopLineLocationIndex, StopRangeBeginDistance + frontBumperDist);
		m_SlowDownBeginIndex = path.rewind(m_StopLineLocationIndex, SlowDownBeginDistance + frontBumperDist, &m_SlowDownBeginDistance);
		m_IndexDelta = m_StopRangeBeginIndex - m_SlowDownBeginIndex;
	}

	UE_LOG(LogDeepDriveTBTStopAtLocationTask, Log, TEXT("DeepDriveTBTStopAtLocationTask::bind %d %d %d"), m_StopLineLocationIndex, m_StopRangeBeginIndex, m_SlowDownBeginIndex);
}

bool DeepDriveTBTStopAtLocationTask::execute(DeepDriveTrafficBlackboard &blackboard, float deltaSeconds, float &speed, int32 pathPointIndex)
{
	if (m_hasStopped == false)
	{
		ADeepDriveAgent *agent = blackboard.getAgent();
		if (agent)
		{
			if (m_SlowDownBeginIndex >= 0 && m_StopRangeBeginIndex >= 0)
			{
				if (pathPointIndex >= m_StopRangeBeginIndex)
				{
					speed = 0.0f;
					const float curSpeed = agent->getSpeedKmh();
					// UE_LOG(LogDeepDriveTBTStopAtLocationTask, Log, TEXT("DeepDriveTBTStopAtLocationTask %d stopping %f"), pathPointIndex, curSpeed);
					if (curSpeed <= 20.0f)
					{
						m_hasStopped = true;
						UE_LOG(LogDeepDriveTBTStopAtLocationTask, Log, TEXT("DeepDriveTBTStopAtLocationTask has Stopped at pathIndex %d"), pathPointIndex);
					}
				}
				else if (pathPointIndex >= m_SlowDownBeginIndex)
				{
					const int32 curIndexDelta = pathPointIndex - m_SlowDownBeginIndex;
					const float curT = static_cast<float> (curIndexDelta) / static_cast<float> (m_IndexDelta);
					const float speedFac = 1.0f - FMath::SmoothStep(0.0f, 0.9f, curT);
					
					// UE_LOG(LogDeepDriveTBTStopAtLocationTask, Log, TEXT("DeepDriveTBTStopAtLocationTask %d slowing down t %f spdFac %f"), pathPointIndex, curT, speedFac);
					speed = FMath::Max(speed * speedFac, 5.0f);
				}
			}
		}
	}

	return !m_hasStopped;
}
