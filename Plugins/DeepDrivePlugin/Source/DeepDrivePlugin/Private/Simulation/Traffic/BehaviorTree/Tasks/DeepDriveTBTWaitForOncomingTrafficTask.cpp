
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
	FVector waitingLocation = blackboard.getVectorValue("WaitingLocation", FVector::ZeroVector);

	const float frontBumperDist = blackboard.getAgent()->getFrontBumperDistance();
	m_WaitingLocationIndex = path.rewind(path.findClosestPathPoint(waitingLocation, 0), frontBumperDist);
	if (m_WaitingLocationIndex >= 0)
	{
		m_SlowDownBeginIndex = path.rewind(m_WaitingLocationIndex, SlowDownBeginDistance);
		m_CheckClearanceIndex = path.rewind(m_WaitingLocationIndex, CheckClearanceDistance);
		m_IndexDelta = m_WaitingLocationIndex - m_SlowDownBeginIndex;
	}

	UE_LOG(LogDeepDriveTBTWaitForOncomingTrafficTask, Log, TEXT("DeepDriveTBTWaitForOncomingTrafficTask::bind %d %d"), m_WaitingLocationIndex, m_SlowDownBeginIndex);
}

bool DeepDriveTBTWaitForOncomingTrafficTask::execute(DeepDriveTrafficBlackboard &blackboard, float deltaSeconds, float &speed, int32 pathPointIndex)
{
	if (m_isJunctionClear == false)
	{
		ADeepDriveAgent *agent = blackboard.getAgent();
		if (agent)
		{
			if	(	m_SlowDownBeginIndex
				&&	pathPointIndex >= m_SlowDownBeginIndex
				)
			{
				const int32 curIndexDelta = pathPointIndex - m_SlowDownBeginIndex;
				const float curT = 1.0f - static_cast<float>(curIndexDelta) / static_cast<float>(m_IndexDelta);
				const float speedFac = FMath::SmoothStep(0.0f, 0.25f, curT);

				m_isJunctionClear = pathPointIndex > m_CheckClearanceIndex ? DeepDriveBehaviorTreeHelpers::isJunctionClear(blackboard) : false;

				if (m_isJunctionClear == false)
				{
					// speed = pathPointIndex < m_WaitingLocationIndex ? FMath::Max(speed * speedFac, SlowDownMinSpeed) : 0.0f;
					speed = pathPointIndex < m_WaitingLocationIndex ? speed * speedFac : 0.0f;
				}

				UE_LOG(LogDeepDriveTBTWaitForOncomingTrafficTask, Log, TEXT("DeepDriveTBTWaitForOncomingTrafficTask spd %f junction clear %c spdFac %f curT %f"), speed, m_isJunctionClear ? 'T' : 'F', speedFac, curT);

			}
		}
	}

	return !m_isJunctionClear;
}
