
#include "Simulation/Traffic/BehaviorTree/Tasks/DeepDriveTBTWaitForOncomingTrafficTask.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBlackboard.h"
#include "Simulation/Agent/DeepDriveAgent.h"

#include "Private/Simulation/Traffic/Path/DeepDrivePartialPath.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveBehaviorTreeHelpers.h"

DEFINE_LOG_CATEGORY(LogDeepDriveTBTWaitForOncomingTrafficTask);

DeepDriveTBTWaitForOncomingTrafficTask::DeepDriveTBTWaitForOncomingTrafficTask()
{
}

void DeepDriveTBTWaitForOncomingTrafficTask::bind(DeepDriveTrafficBlackboard &blackboard, DeepDrivePartialPath &path)
{
	FVector stopLineLocation = blackboard.getVectorValue("StopLineLocation", FVector::ZeroVector);

	m_StopLineLocationIndex = path.findClosestPathPoint(stopLineLocation, 0);
	if(m_StopLineLocationIndex >= 0)
	{
		const float frontBumperDist = blackboard.getAgent()->getFrontBumperDistance();
		m_StopRangeBeginIndex = path.rewind(m_StopLineLocationIndex, StopRangeBeginDistance + frontBumperDist);
		m_SlowDownBeginIndex = path.rewind(m_StopLineLocationIndex, SlowDownBeginDistance + frontBumperDist, &m_SlowDownBeginDistance);
		m_CheckClearanceIndex = path.rewind(m_StopLineLocationIndex, SlowDownBeginDistance + frontBumperDist);
		m_IndexDelta = m_StopRangeBeginIndex - m_SlowDownBeginIndex;
	}

	UE_LOG(LogDeepDriveTBTWaitForOncomingTrafficTask, Log, TEXT("DeepDriveTBTWaitForOncomingTrafficTask::bind %d %d %d"), m_StopLineLocationIndex, m_StopRangeBeginIndex, m_SlowDownBeginIndex);
}

bool DeepDriveTBTWaitForOncomingTrafficTask::execute(DeepDriveTrafficBlackboard &blackboard, float deltaSeconds, float &speed, int32 pathPointIndex)
{
	bool res = true;
	SDeepDriveManeuver *maneuver = blackboard.getManeuver();
	if(maneuver->ManeuverType == EDeepDriveManeuverType::TURN_LEFT)
	{

	}

	return res;
	
#if 0
	if (m_hasReachedStopLine == false)
	{
		ADeepDriveAgent *agent = blackboard.getAgent();
		if (agent)
		{
			if (m_SlowDownBeginIndex >= 0 && m_StopRangeBeginIndex >= 0)
			{
				
				if(pathPointIndex >= m_CheckClearanceIndex)
				{
					m_isJunctionClear = DeepDriveBehaviorTreeHelpers::isJunctionClear(blackboard);
				}

				if (pathPointIndex >= m_StopRangeBeginIndex)
				{
					if(m_isJunctionClear)
						m_hasReachedStopLine = true;
					else
						speed = 0.0f;
				}
				else if (pathPointIndex >= m_SlowDownBeginIndex)
				{
					const int32 curIndexDelta = pathPointIndex - m_SlowDownBeginIndex;
					const float curT = static_cast<float> (curIndexDelta) / static_cast<float> (m_IndexDelta);
					const float speedFac = 1.0f - FMath::SmoothStep(0.0f, 0.9f, curT);
					
					// UE_LOG(LogDeepDriveTBTWaitForOncomingTrafficTask, Log, TEXT("DeepDriveTBTWaitForOncomingTrafficTask %d slowing down t %f spdFac %f"), pathPointIndex, curT, speedFac);
					speed = FMath::Max(speed * speedFac, SlowDownMinSpeed);
				}
			}
		}
	}

	return !m_hasReachedStopLine;
#endif
}
