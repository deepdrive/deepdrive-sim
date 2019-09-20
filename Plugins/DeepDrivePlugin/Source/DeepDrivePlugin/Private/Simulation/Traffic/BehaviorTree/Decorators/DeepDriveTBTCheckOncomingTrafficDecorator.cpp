
#include "Simulation/Traffic/BehaviorTree/Decorators/DeepDriveTBTCheckOncomingTrafficDecorator.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBlackboard.h"
#include "Simulation/Agent/DeepDriveAgent.h"

#include "Private/Simulation/Traffic/Path/DeepDrivePartialPath.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveBehaviorTreeHelpers.h"

DEFINE_LOG_CATEGORY(LogDeepDriveTBTCheckOncomingTrafficDecorator);

DeepDriveTBTCheckOncomingTrafficDecorator::DeepDriveTBTCheckOncomingTrafficDecorator(const FString &refLocName)
{
	m_ReferenceLocationName = refLocName;
}

void DeepDriveTBTCheckOncomingTrafficDecorator::bind(DeepDriveTrafficBlackboard &blackboard, DeepDrivePartialPath &path)
{
	FVector referenceLocation = blackboard.getVectorValue(m_ReferenceLocationName, FVector::ZeroVector);

	const float frontBumperDist = blackboard.getAgent()->getFrontBumperDistance();
	m_CheckClearanceIndex = path.rewind(path.findClosestPathPoint(referenceLocation, 0), frontBumperDist + CheckClearanceDistance);

	UE_LOG(LogDeepDriveTBTCheckOncomingTrafficDecorator, Log, TEXT("LogDeepDriveTBTCheckOncomingTrafficDecorator::bind %d"), m_CheckClearanceIndex);
}

bool DeepDriveTBTCheckOncomingTrafficDecorator::performCheck(DeepDriveTrafficBlackboard &blackboard, int32 pathPointIndex)
{
	bool isJunctionClear = false;
	if(pathPointIndex > m_CheckClearanceIndex)
	{
		ADeepDriveAgent *agent = blackboard.getAgent();
		if (agent)
		{
			isJunctionClear = DeepDriveBehaviorTreeHelpers::isJunctionClear(blackboard);

			// UE_LOG(LogDeepDriveTBTCheckOncomingTrafficDecorator, Log, TEXT("LogDeepDriveTBTCheckOncomingTrafficDecorator [%d] junction clear %c"), agent->GetAgentId(), isJunctionClear ? 'T' : 'F');
		}
	}
	return !isJunctionClear;
}
