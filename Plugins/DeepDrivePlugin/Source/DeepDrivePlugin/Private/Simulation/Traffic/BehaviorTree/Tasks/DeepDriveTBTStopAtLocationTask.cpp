
#include "Simulation/Traffic/BehaviorTree/Tasks/DeepDriveTBTStopAtLocationTask.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBlackboard.h"
#include "Simulation/Agent/DeepDriveAgent.h"

#include "Private/Simulation/Traffic/Path/DeepDrivePartialPath.h"

DEFINE_LOG_CATEGORY(LogDeepDriveTBTStopAtLocationTask);

DeepDriveTBTStopAtLocationTask::DeepDriveTBTStopAtLocationTask(const FString &stopLocationName, float exponent, float stopBeginDistance, float slowDownBeginDistance)
	:	m_StopLocationName(stopLocationName)
	,	m_Exponent(exponent)
	,	StopBeginDistance(stopBeginDistance)
	,	SlowDownBeginDistance(slowDownBeginDistance)
{
}

void DeepDriveTBTStopAtLocationTask::bind(DeepDriveTrafficBlackboard &blackboard, DeepDrivePartialPath &path)
{
	if(m_StopLocationIndex < 0)
	{
		FVector stopLocation = blackboard.getVectorValue(m_StopLocationName, FVector::ZeroVector);

		const float frontBumperDist = blackboard.getAgent()->getFrontBumperDistance();
		m_StopLocationIndex = path.rewind(path.findClosestPathPoint(stopLocation, 0), frontBumperDist);
		if (m_StopLocationIndex >= 0)
		{
			m_StopBeginIndex = path.rewind(m_StopLocationIndex, StopBeginDistance);
			m_SlowDownBeginIndex = path.rewind(m_StopLocationIndex, SlowDownBeginDistance);
			m_IndexDelta = m_StopLocationIndex - m_SlowDownBeginIndex;
		}

		UE_LOG(LogDeepDriveTBTStopAtLocationTask, Log, TEXT("DeepDriveTBTStopAtLocationTask::bind [%d] %s (%s) %d %d"), blackboard.getAgent()->GetAgentId(), *m_StopLocationName, *(stopLocation.ToString()), m_StopLocationIndex, m_SlowDownBeginIndex);
	}
}

bool DeepDriveTBTStopAtLocationTask::execute(DeepDriveTrafficBlackboard &blackboard, float deltaSeconds, float &speed, int32 pathPointIndex)
{
	bool hasStopped = false;
	ADeepDriveAgent *agent = blackboard.getAgent();
	if (agent)
	{
		if (m_SlowDownBeginIndex >= 0 && pathPointIndex >= m_SlowDownBeginIndex)
		{
			const int32 curIndexDelta = pathPointIndex - m_SlowDownBeginIndex;
			const float curT = FMath::Clamp(1.0f - static_cast<float>(curIndexDelta) / static_cast<float>(m_IndexDelta), 0.0f, 1.0f);
			const float speedFac = FMath::Pow(curT, m_Exponent);

			speed = pathPointIndex < m_StopLocationIndex ? speed * speedFac : 0.0f;
			hasStopped = pathPointIndex >= m_StopBeginIndex && agent->getSpeedKmh() < 1.0f;

			// UE_LOG(LogDeepDriveTBTStopAtLocationTask, Log, TEXT("DeepDriveTBTStopAtLocationTask[%p] %d spd %f spdFac %f curT %f agntSpd %f -> %c"), this, pathPointIndex, speed, speedFac, curT, agent->getSpeedKmh(), m_hasStopped ? 'T' : 'F');
		}
	}

	return !hasStopped;
}
