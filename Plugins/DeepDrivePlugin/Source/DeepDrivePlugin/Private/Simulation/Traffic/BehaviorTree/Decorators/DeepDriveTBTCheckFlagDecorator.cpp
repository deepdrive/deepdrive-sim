
#include "Simulation/Traffic/BehaviorTree/Decorators/DeepDriveTBTCheckFlagDecorator.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBlackboard.h"
#include "Simulation/Traffic/Path/DeepDrivePathDefines.h"
#include "Simulation/Agent/DeepDriveAgent.h"
#include "Simulation/Traffic/Path/DeepDrivePartialPath.h"

DeepDriveTBTCheckFlagDecorator::DeepDriveTBTCheckFlagDecorator(const FString &flagName, bool refValue, bool defaultValue)
	:	m_FlagName(flagName)
	,	m_RefValue(refValue)
	,	m_DefaultValue(defaultValue)
{

}

bool DeepDriveTBTCheckFlagDecorator::performCheck(DeepDriveTrafficBlackboard &blackboard, int32 pathPointIndex) const
{
	const bool res = blackboard.getBooleanValue(m_FlagName, m_DefaultValue) == m_RefValue;
	return res;
}
