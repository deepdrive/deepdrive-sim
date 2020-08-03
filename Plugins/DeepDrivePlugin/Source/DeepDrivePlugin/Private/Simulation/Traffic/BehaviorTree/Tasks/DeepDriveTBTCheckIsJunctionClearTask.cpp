
#include "Simulation/Traffic/BehaviorTree/Tasks/DeepDriveTBTCheckIsJunctionClearTask.h"

#include "Simulation/Traffic/BehaviorTree/DeepDriveBehaviorTreeHelpers.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBlackboard.h"

#include "Simulation/Agent/DeepDriveAgent.h"
#include "Private/Simulation/Traffic/Path/DeepDrivePartialPath.h"

#include "ActorEventLogging/Public/ActorEventLoggingMacros.h"

DEFINE_LOG_CATEGORY(LogDeepDriveTBTCheckIsJunctionClearTask);

DeepDriveTBTCheckIsJunctionClearTask::DeepDriveTBTCheckIsJunctionClearTask(const FString &refLocationName, float distance, bool ignoreTrafficLights)
	:	m_RefLocationName(refLocationName)
	,	m_Distance(distance)
	,	m_ignoreTrafficLights(false)
	,	m_FlagName()
	,	m_RefLocationIndex(-1)
{

}

DeepDriveTBTCheckIsJunctionClearTask::DeepDriveTBTCheckIsJunctionClearTask(const FString &refLocationName, float distance, bool ignoreTrafficLights, const FString &flagName)
	:	m_RefLocationName(refLocationName)
	,	m_Distance(distance)
	,	m_ignoreTrafficLights(ignoreTrafficLights)
	,	m_FlagName(flagName)
	,	m_RefLocationIndex(-1)
{

}

void DeepDriveTBTCheckIsJunctionClearTask::bind(DeepDriveTrafficBlackboard &blackboard, DeepDrivePartialPath &path)
{
	if(m_RefLocationIndex == -1)
	{
		FVector refLocation = blackboard.getVectorValue(m_RefLocationName, FVector::ZeroVector);

		const float frontBumperDist = blackboard.getAgent()->getFrontBumperDistance();
		m_RefLocationIndex = path.rewind(path.findClosestPathPoint(refLocation, 0), frontBumperDist + m_Distance);

		UE_LOG(LogDeepDriveTBTCheckIsJunctionClearTask, Log, TEXT("LogDeepDriveTBTCheckIsJunctionClearTask::bind [%d] %d -> %s"), blackboard.getAgent()->GetAgentId(), m_RefLocationIndex, *m_FlagName);
	}
}

bool DeepDriveTBTCheckIsJunctionClearTask::execute(DeepDriveTrafficBlackboard &blackboard, float deltaSeconds, float &speed, int32 pathPointIndex)
{
	if(pathPointIndex >= m_RefLocationIndex)
	{
		const float junctionClearValue = DeepDriveBehaviorTreeHelpers::calculateJunctionClearValue(blackboard, m_ignoreTrafficLights);
		const bool isJunctionClear = junctionClearValue > 0.9f;

		if(m_FlagName.IsEmpty() == false)
			blackboard.setBooleanValue(m_FlagName, isJunctionClear);

		// UE_LOG(LogDeepDriveTBTCheckIsJunctionClearTask, Log, TEXT("LogDeepDriveTBTCheckIsJunctionClearTask::execute [%d] %d %c"), blackboard.getAgent()->GetAgentId(), pathPointIndex, isJunctionClear ? 'T' : 'F');
		AEL_MESSAGE((*(blackboard.getAgent())), TEXT("CheckIsJunctionClearTask::execute %d %c"), pathPointIndex, isJunctionClear ? 'T' : 'F');
	}

	return true;
}
