
#include "Simulation/Traffic/BehaviorTree/Decorators/DeepDriveTBTIsInRangeDecorator.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBlackboard.h"
#include "Simulation/Traffic/Path/DeepDrivePathDefines.h"
#include "Simulation/Agent/DeepDriveAgent.h"
#include "Private/Simulation/Traffic/Path/DeepDrivePartialPath.h"

DeepDriveTBTIsInRangeDecorator::DeepDriveTBTIsInRangeDecorator(const FString refLocationName, float range)
	:	m_ReferenceLocationName(refLocationName)
	,	m_Range(range)
{

}

void DeepDriveTBTIsInRangeDecorator::bind(DeepDriveTrafficBlackboard &blackboard, DeepDrivePartialPath &path)
{
	FVector referenceLocation = blackboard.getVectorValue(m_ReferenceLocationName, FVector::ZeroVector);

	const float frontBumperDist = blackboard.getAgent()->getFrontBumperDistance();
	m_RangeEndLocationIndex = path.rewind(path.findClosestPathPoint(referenceLocation, 0), frontBumperDist);
	m_RangeBeginLocationIndex = path.rewind(m_RangeEndLocationIndex, m_Range);
}

bool DeepDriveTBTIsInRangeDecorator::performCheck(DeepDriveTrafficBlackboard &blackboard, int32 pathPointIndex)
{
	const bool res = pathPointIndex >= m_RangeBeginLocationIndex && pathPointIndex <= m_RangeEndLocationIndex;
	return res;
}
