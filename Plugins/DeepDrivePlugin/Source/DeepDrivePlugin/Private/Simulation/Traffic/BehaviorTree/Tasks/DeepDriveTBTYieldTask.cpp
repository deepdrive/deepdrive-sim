
#include "Simulation/Traffic/BehaviorTree/Tasks/DeepDriveTBTYieldTask.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBlackboard.h"
#include "Simulation/Agent/DeepDriveAgent.h"

#include "Private/Simulation/Traffic/Path/DeepDrivePartialPath.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveBehaviorTreeHelpers.h"

#include "Public/Simulation/DeepDriveSimulation.h"

DEFINE_LOG_CATEGORY(LogDeepDriveTBTYieldTask);

DeepDriveTBTYieldTask::DeepDriveTBTYieldTask()
{
}

void DeepDriveTBTYieldTask::bind(DeepDriveTrafficBlackboard &blackboard, DeepDrivePartialPath &path)
{
	FVector lineOfSightLocation = blackboard.getVectorValue("LineOfSightLocation", FVector::ZeroVector);

	const float frontBumperDist = blackboard.getAgent()->getFrontBumperDistance();
	m_LineOfSightLocationIndex = path.rewind(path.findClosestPathPoint(lineOfSightLocation, 0), frontBumperDist);
	if (m_LineOfSightLocationIndex >= 0)
	{
		m_StopRangeBeginIndex = path.rewind(m_LineOfSightLocationIndex, StopRangeBeginDistance);
		m_SlowDownBeginIndex = path.rewind(m_LineOfSightLocationIndex, SlowDownBeginDistance, &m_SlowDownBeginDistance);
		m_CheckClearanceIndex = path.rewind(m_LineOfSightLocationIndex, CheckClearanceDistance);
		m_PastStopPointIndex = path.windForward(m_LineOfSightLocationIndex, PastStopPointDistance);
		m_IndexDelta = m_StopRangeBeginIndex - m_SlowDownBeginIndex;
	}

	UE_LOG(LogDeepDriveTBTYieldTask, Log, TEXT("DeepDriveTBTYieldTask::bind %d %d %d"), m_LineOfSightLocationIndex, m_StopRangeBeginIndex, m_SlowDownBeginIndex);
}

#if 1
bool DeepDriveTBTYieldTask::execute(DeepDriveTrafficBlackboard &blackboard, float deltaSeconds, float &speed, int32 pathPointIndex)
{
	if (m_hasReachedStopLine == false)
	{
		ADeepDriveAgent *agent = blackboard.getAgent();
		if (agent)
		{
			if (m_SlowDownBeginIndex >= 0 && m_StopRangeBeginIndex >= 0)
			{
				if (pathPointIndex >= m_CheckClearanceIndex)
				{
					if (pathPointIndex <= m_PastStopPointIndex)
					{
						float junctionClearValue = DeepDriveBehaviorTreeHelpers::calculateJunctionClearValue(blackboard);
						UE_LOG(LogDeepDriveTBTYieldTask, Log, TEXT("DeepDriveTBTYieldTask[%d|%d] clear value %f"), pathPointIndex, m_CheckClearanceIndex, junctionClearValue);
						speed *= junctionClearValue;
					}
					else
						UE_LOG(LogDeepDriveTBTYieldTask, Log, TEXT("DeepDriveTBTYieldTask[%d|%d] Too late, Keep going"), pathPointIndex, m_PastStopPointIndex);
				}
				else if (pathPointIndex >= m_SlowDownBeginIndex)
				{
					const int32 curIndexDelta = pathPointIndex - m_SlowDownBeginIndex;
					const float curT = static_cast<float>(curIndexDelta) / static_cast<float>(m_IndexDelta);
					const float speedFac = 1.0f - FMath::SmoothStep(0.0f, 0.9f, curT);

					// UE_LOG(LogDeepDriveTBTYieldTask, Log, TEXT("DeepDriveTBTYieldTask %d slowing down t %f spdFac %f"), pathPointIndex, curT, speedFac);
					speed = FMath::Max(speed * speedFac, SlowDownMinSpeed);
				}
			}
		}
	}

	return !m_hasReachedStopLine;
}

#else

bool DeepDriveTBTYieldTask::execute(DeepDriveTrafficBlackboard &blackboard, float deltaSeconds, float &speed, int32 pathPointIndex)
{
	if (m_hasReachedStopLine == false)
	{
		ADeepDriveAgent *agent = blackboard.getAgent();
		if (agent)
		{
			if (m_SlowDownBeginIndex >= 0 && m_StopRangeBeginIndex >= 0)
			{

				if (pathPointIndex >= m_CheckClearanceIndex)
				{
					float junctionClearValue = DeepDriveBehaviorTreeHelpers::calculateJunctionClearValue(blackboard);
					m_isJunctionClear = DeepDriveBehaviorTreeHelpers::isJunctionClear(blackboard);
					UE_LOG(LogDeepDriveTBTYieldTask, Log, TEXT("DeepDriveTBTYieldTask clear value %f"), junctionClearValue);
				}

				if (pathPointIndex >= m_LineOfSightLocationIndex)
				{
					if (m_isJunctionClear)
						m_hasReachedStopLine = true;
					else
						speed = 0.0f;
				}
				else if (pathPointIndex >= m_SlowDownBeginIndex)
				{
					const int32 curIndexDelta = pathPointIndex - m_SlowDownBeginIndex;
					const float curT = static_cast<float>(curIndexDelta) / static_cast<float>(m_IndexDelta);
					const float speedFac = 1.0f - FMath::SmoothStep(0.0f, 0.9f, curT);

					// UE_LOG(LogDeepDriveTBTYieldTask, Log, TEXT("DeepDriveTBTYieldTask %d slowing down t %f spdFac %f"), pathPointIndex, curT, speedFac);
					speed = FMath::Max(speed * speedFac, SlowDownMinSpeed);
				}
			}
		}
	}

	return !m_hasReachedStopLine;
}
#endif
