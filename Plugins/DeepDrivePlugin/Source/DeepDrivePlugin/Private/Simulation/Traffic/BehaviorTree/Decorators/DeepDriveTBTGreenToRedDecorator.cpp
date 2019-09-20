
#include "Simulation/Traffic/BehaviorTree/Decorators/DeepDriveTBTGreenToRedDecorator.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBlackboard.h"
#include "Simulation/TrafficLight/DeepDriveTrafficLight.h"
#include "Simulation/Traffic/Path/DeepDrivePathDefines.h"
#include "Simulation/Agent/DeepDriveAgent.h"
#include "Private/Simulation/Traffic/Path/DeepDrivePartialPath.h"

DEFINE_LOG_CATEGORY(LogDeepDriveTBTGreenToRedDecorator);

void DeepDriveTBTGreenToRedDecorator::bind(DeepDriveTrafficBlackboard &blackboard, DeepDrivePartialPath &path)
{
	FVector referenceLocation = blackboard.getVectorValue("StopLineLocation", FVector::ZeroVector);

	const float frontBumperDist = blackboard.getAgent()->getFrontBumperDistance();
	m_StopLineLocationIndex = path.rewind(path.findClosestPathPoint(referenceLocation, 0), frontBumperDist);
}

bool DeepDriveTBTGreenToRedDecorator::performCheck(DeepDriveTrafficBlackboard &blackboard, int32 pathPointIndex)
{
	bool res = false;

	SDeepDriveManeuver *maneuver = blackboard.getManeuver();
	ADeepDriveAgent *agent = blackboard.getAgent();
	if	(	agent && maneuver && maneuver->TrafficLight
		&&	maneuver->TrafficLight->CurrentPhase == EDeepDriveTrafficLightPhase::GREEN_TO_RED
		)
	{
		// status: -1 not set, 0 stop, 1 go on
		int32 status = blackboard.getIntegerValue("GreenToRedStatus", -1);
		if(status < 0)
		{
			const float dist = blackboard.getPartialPath()->getDistance(pathPointIndex, m_StopLineLocationIndex);
			const float remainingGreenTime = maneuver->TrafficLight->getRemainingPhaseTime();
			const float curSpeed = agent->getSpeed();
			const float coveredDist = curSpeed * remainingGreenTime * 0.9f;

			status = coveredDist <= dist ? 0 : 1;	// if there is no chance of making it, stop

			blackboard.setIntegerValue("GreenToRedStatus", status);

			UE_LOG(LogDeepDriveTBTGreenToRedDecorator, Log, TEXT("Distance to StopLine at %d status %d  | %f <-> %f"), pathPointIndex, status, coveredDist, dist);
		}

		res = status > 0 ? false : true;
	}

	return res;
}
