
#include "Simulation/Traffic/BehaviorTree/Decorators/DeepDriveTBTCheckIntDecorator.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBlackboard.h"
#include "Simulation/Traffic/Path/DeepDrivePathDefines.h"
#include "Simulation/Agent/DeepDriveAgent.h"
#include "Private/Simulation/Traffic/Path/DeepDrivePartialPath.h"

DeepDriveTBTCheckIntDecorator::DeepDriveTBTCheckIntDecorator(const FString &flagName, int32 refValue, int32 defaultValue)
	:	m_FlagName(flagName)
	,	m_RefValue(refValue)
	,	m_DefaultValue(defaultValue)
{

}

bool DeepDriveTBTCheckIntDecorator::performCheck(DeepDriveTrafficBlackboard &blackboard, int32 pathPointIndex) const
{
	const bool res = blackboard.getIntegerValue(m_FlagName, m_DefaultValue) == m_RefValue;
	return res;
}
