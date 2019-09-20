
#include "Simulation/Traffic/BehaviorTree/Decorators/DeepDriveTBTCheckTrafficLightDecorator.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBlackboard.h"
#include "Simulation/TrafficLight/DeepDriveTrafficLight.h"
#include "Simulation/Traffic/Path/DeepDrivePathDefines.h"

DeepDriveTBTCheckTrafficLightDecorator::DeepDriveTBTCheckTrafficLightDecorator(EDeepDriveTrafficLightPhase phase)
	:	m_Phase(phase)
{

}

bool DeepDriveTBTCheckTrafficLightDecorator::performCheck(DeepDriveTrafficBlackboard &blackboard, int32 pathPointIndex)
{
	SDeepDriveManeuver *maneuver = blackboard.getManeuver();
	const bool res = maneuver && maneuver->TrafficLight && maneuver->TrafficLight->CurrentPhase == m_Phase;

	return res;
}
