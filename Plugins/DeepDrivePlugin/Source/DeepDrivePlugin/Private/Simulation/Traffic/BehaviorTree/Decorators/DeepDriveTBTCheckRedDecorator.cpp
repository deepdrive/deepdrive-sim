
#include "Simulation/Traffic/BehaviorTree/Decorators/DeepDriveTBTCheckRedDecorator.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBlackboard.h"
#include "Simulation/TrafficLight/DeepDriveTrafficLight.h"
#include "Simulation/Traffic/Path/DeepDrivePathDefines.h"

bool DeepDriveTBTCheckRedDecorator::performCheck(DeepDriveTrafficBlackboard &blackboard, int32 pathPointIndex)
{
	const int32 greenToRedStatus = blackboard.getIntegerValue("GreenToRedStatus", -1);
	SDeepDriveManeuver *maneuver = blackboard.getManeuver();
	const bool res =	maneuver && maneuver->TrafficLight
					&&	maneuver->TrafficLight->CurrentPhase == EDeepDriveTrafficLightPhase::RED
					&&  greenToRedStatus <= 0;

	return res;
}
