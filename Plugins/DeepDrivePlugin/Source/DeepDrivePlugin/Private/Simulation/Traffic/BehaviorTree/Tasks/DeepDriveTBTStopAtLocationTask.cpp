
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
	FVector stopLocation = blackboard.getVectorValue("StopLineLocation", FVector::ZeroVector);

	const float frontBumperDist = blackboard.getAgent()->getFrontBumperDistance();
	m_StopLocationIndex = path.rewind(path.findClosestPathPoint(stopLocation, 0), frontBumperDist);
	if (m_StopLocationIndex >= 0)
	{
		m_StopBeginIndex = path.rewind(m_StopLocationIndex, StopBeginDistance);
		m_SlowDownBeginIndex = path.rewind(m_StopLocationIndex, SlowDownBeginDistance);
		m_IndexDelta = m_StopLocationIndex - m_SlowDownBeginIndex;
	}

	UE_LOG(LogDeepDriveTBTStopAtLocationTask, Log, TEXT("DeepDriveTBTStopAtLocationTask::bind %d %d | %s"), m_StopLocationIndex, m_SlowDownBeginIndex, *(stopLocation.ToString()));
}

bool DeepDriveTBTStopAtLocationTask::execute(DeepDriveTrafficBlackboard &blackboard, float deltaSeconds, float &speed, int32 pathPointIndex)
{
	if (m_hasStopped == false)
	{
		ADeepDriveAgent *agent = blackboard.getAgent();
		if (agent)
		{
			if (m_SlowDownBeginIndex && pathPointIndex >= m_SlowDownBeginIndex)
			{
				const int32 curIndexDelta = pathPointIndex - m_SlowDownBeginIndex;
				const float curT = 1.0f - static_cast<float>(curIndexDelta) / static_cast<float>(m_IndexDelta);
				const float speedFac = FMath::SmoothStep(0.0f, 0.5f, curT);

				speed = pathPointIndex < m_StopLocationIndex ? speed * speedFac : 0.0f;
				m_hasStopped = pathPointIndex > m_StopBeginIndex && agent->getSpeedKmh() < 1.0f;

				//UE_LOG(LogDeepDriveTBTStopAtLocationTask, Log, TEXT("DeepDriveTBTStopAtLocationTask spd %f spdFac %f curT %f agntSpd %f -> %c"), speed, speedFac, curT, agent->getSpeedKmh(), m_hasStopped ? 'T' : 'F');
			}
		}
	}

	return !m_hasStopped;
}
