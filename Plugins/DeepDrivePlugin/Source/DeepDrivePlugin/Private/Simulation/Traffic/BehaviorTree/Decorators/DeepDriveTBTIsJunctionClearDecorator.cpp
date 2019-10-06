
#include "Simulation/Traffic/BehaviorTree/Decorators/DeepDriveTBTIsJunctionClearDecorator.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBlackboard.h"

#include "Simulation/Traffic/BehaviorTree/DeepDriveBehaviorTreeHelpers.h"

DEFINE_LOG_CATEGORY(LogDeepDriveTBTIsJunctionClearDecorator);

DeepDriveTBTIsJunctionClearDecorator::DeepDriveTBTIsJunctionClearDecorator(bool checkOnce)
	:	m_CheckOnce(checkOnce)
{

}

bool DeepDriveTBTIsJunctionClearDecorator::performCheck(DeepDriveTrafficBlackboard &blackboard, int32 pathPointIndex)
{
	if	(!m_CheckOnce || !m_isJunctionClear)
	{
		const float junctionClearValue = DeepDriveBehaviorTreeHelpers::calculateJunctionClearValue(blackboard);
		m_isJunctionClear = junctionClearValue > 0.9f;
		UE_LOG(LogDeepDriveTBTIsJunctionClearDecorator, Log, TEXT("[%d] %f"), blackboard.getIntegerValue("AgentId"), junctionClearValue);
	}

	return !m_isJunctionClear;
}
