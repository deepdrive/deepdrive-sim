
#include "Simulation/Traffic/BehaviorTree/Tasks/DeepDriveTBTCheckGreenToRedTask.h"

#include "Simulation/Traffic/BehaviorTree/DeepDriveBehaviorTreeHelpers.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBlackboard.h"

#include "Simulation/Agent/DeepDriveAgent.h"
#include "Private/Simulation/Traffic/Path/DeepDrivePartialPath.h"
#include "Simulation/TrafficLight/DeepDriveTrafficLight.h"


DEFINE_LOG_CATEGORY(LogDeepDriveTBTCheckGreenToRedTask);

DeepDriveTBTCheckGreenToRedTask::DeepDriveTBTCheckGreenToRedTask(const FString &refLocationName, const FString &flagName)
	:	m_RefLocationName(refLocationName)
	,	m_FlagName(flagName)
	,	m_RefLocationIndex(-1)
{
}

void DeepDriveTBTCheckGreenToRedTask::bind(DeepDriveTrafficBlackboard &blackboard, DeepDrivePartialPath &path)
{
	FVector refLocation = blackboard.getVectorValue(m_RefLocationName, FVector::ZeroVector);

	const float frontBumperDist = blackboard.getAgent()->getFrontBumperDistance();
	m_RefLocationIndex = path.rewind(path.findClosestPathPoint(refLocation, 0), frontBumperDist);
	UE_LOG(LogDeepDriveTBTCheckGreenToRedTask, Log, TEXT("LogDeepDriveTBTCheckGreenToRedTask::bind [%d] %d"), blackboard.getAgent()->GetAgentId(), m_RefLocationIndex);
}

bool DeepDriveTBTCheckGreenToRedTask::execute(DeepDriveTrafficBlackboard &blackboard, float deltaSeconds, float &speed, int32 pathPointIndex)
{
	SDeepDriveManeuver *maneuver = blackboard.getManeuver();
	ADeepDriveAgent *agent = blackboard.getAgent();
	if	(	agent && maneuver && maneuver->TrafficLight
		&&	blackboard.getIntegerValue(m_FlagName, -1) < 0
		)
	{
		const float dist = blackboard.getPartialPath()->getDistance(pathPointIndex, m_RefLocationIndex);
		const float remainingGreenTime = maneuver->TrafficLight->getRemainingPhaseTime();
		const float curSpeed = agent->getSpeed();
		const float coveredDist = curSpeed * remainingGreenTime * 0.9f;

		const int32 status = coveredDist <= dist ? 0 : 1;	// if there is no chance of making it, stop
		UE_LOG(LogDeepDriveTBTCheckGreenToRedTask, Log, TEXT("LogDeepDriveTBTCheckGreenToRedTask::execute [%d] Status %d"), blackboard.getAgent()->GetAgentId(), status);

		blackboard.setIntegerValue(m_FlagName, status);
		return true;
	}
	return false;
}
