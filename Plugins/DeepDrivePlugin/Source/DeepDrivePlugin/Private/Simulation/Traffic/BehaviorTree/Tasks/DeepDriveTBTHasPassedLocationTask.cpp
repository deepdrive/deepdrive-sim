
#include "Simulation/Traffic/BehaviorTree/Tasks/DeepDriveTBTHasPassedLocationTask.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBlackboard.h"
#include "Simulation/Agent/DeepDriveAgent.h"

#include "Simulation/Traffic/Path/DeepDrivePartialPath.h"

DEFINE_LOG_CATEGORY(LogDeepDriveTBTHasPassedLocationTask);

DeepDriveTBTHasPassedLocationTask::DeepDriveTBTHasPassedLocationTask(const FString &locationName, float distance, const FString &flagName)
	:	m_LocationName(locationName)
	,	m_Distance(distance)
	,	m_FlagName(flagName)
{
}

void DeepDriveTBTHasPassedLocationTask::bind(DeepDriveTrafficBlackboard &blackboard, DeepDrivePartialPath &path)
{
	if(m_LocationIndex < 0)
	{
		FVector location = blackboard.getVectorValue(m_LocationName, FVector::ZeroVector);

		const float frontBumperDist = blackboard.getAgent()->getFrontBumperDistance();
		m_LocationIndex = path.rewind(path.findClosestPathPoint(location, 0), frontBumperDist);
		m_LocationIndex = path.windForward(m_LocationIndex, m_Distance);

		UE_LOG(LogDeepDriveTBTHasPassedLocationTask, Log, TEXT("DeepDriveTBTHasPassedLocationTask::bind [%d] %s (%s) %d"), blackboard.getAgent()->GetAgentId(), *m_LocationName, *(location.ToString()), m_LocationIndex);
	}
}

bool DeepDriveTBTHasPassedLocationTask::execute(DeepDriveTrafficBlackboard &blackboard, float deltaSeconds, float &speed, int32 pathPointIndex)
{
	if(m_LocationIndex >= 0 && pathPointIndex >= m_LocationIndex)
	{
		blackboard.setBooleanValue(m_FlagName, true);
		UE_LOG(LogDeepDriveTBTHasPassedLocationTask, Log, TEXT("DeepDriveTBTHasPassedLocationTask::execute [%d] has passed %s "), blackboard.getAgent()->GetAgentId(), *m_LocationName);
	}

	return true;
}
