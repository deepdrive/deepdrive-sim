
#pragma once

#include "CoreMinimal.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveManeuverCalculator, Log, All);

struct SDeepDriveRoadNetwork;
struct SDeepDriveRoute;
struct SDeepDriveManeuver;
class ADeepDriveAgent;
class ADeepDriveSimulation;

class DeepDriveJunctionCalculatorBase;

class DeepDriveManeuverCalculator
{
	typedef TMap<uint32, DeepDriveJunctionCalculatorBase*>	TJunctionCalculators;

public:

	DeepDriveManeuverCalculator(const SDeepDriveRoadNetwork &roadNetwork, ADeepDriveSimulation &simulation);

	void calculate(SDeepDriveRoute &route, ADeepDriveAgent &agent);


private:

	void extractCrossRoadTraffic(SDeepDriveManeuver &maneuver);

	void calculateFakedManeuver(SDeepDriveRoute &route, ADeepDriveAgent &agent);

	const SDeepDriveRoadNetwork				&m_RoadNetwork;

	ADeepDriveSimulation					&m_Simulation;

	TJunctionCalculators					m_JunctionCalculators;
	
};
