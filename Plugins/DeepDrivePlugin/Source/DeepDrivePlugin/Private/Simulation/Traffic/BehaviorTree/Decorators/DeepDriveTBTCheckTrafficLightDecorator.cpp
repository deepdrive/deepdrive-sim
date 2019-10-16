
#include "Simulation/Traffic/BehaviorTree/Decorators/DeepDriveTBTCheckTrafficLightDecorator.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBlackboard.h"
#include "Simulation/TrafficLight/DeepDriveTrafficLight.h"
#include "Simulation/Traffic/Path/DeepDrivePathDefines.h"

DEFINE_LOG_CATEGORY(LogDeepDriveTBTCheckTrafficLightDecorator);

DeepDriveTBTCheckTrafficLightDecorator::DeepDriveTBTCheckTrafficLightDecorator(EDeepDriveTrafficLightPhase mainPhase)
	:	m_MainPhase(mainPhase)
	,	m_AlternativePhase(EDeepDriveTrafficLightPhase::UNDEFINED)
{

}

DeepDriveTBTCheckTrafficLightDecorator::DeepDriveTBTCheckTrafficLightDecorator(EDeepDriveTrafficLightPhase mainPhase, EDeepDriveTrafficLightPhase altPhase)
	:	m_MainPhase(mainPhase)
	,	m_AlternativePhase(altPhase)
{

}

bool DeepDriveTBTCheckTrafficLightDecorator::performCheck(DeepDriveTrafficBlackboard &blackboard, int32 pathPointIndex) const
{
	SDeepDriveManeuver *maneuver = blackboard.getManeuver();
	const EDeepDriveTrafficLightPhase curPhase = maneuver && maneuver->TrafficLight ? maneuver->TrafficLight->CurrentPhase : EDeepDriveTrafficLightPhase::UNDEFINED;

	const bool res =	m_AlternativePhase == EDeepDriveTrafficLightPhase::UNDEFINED
					?	curPhase == m_MainPhase
					:	curPhase == m_MainPhase || curPhase == m_AlternativePhase
					;

	// UE_LOG(LogDeepDriveTBTCheckTrafficLightDecorator, Log, TEXT(">>>>>>>>> Execute %c  curPhase %d"), res ? 'T' : 'F', static_cast<int32> (curPhase));

	return res;
}
