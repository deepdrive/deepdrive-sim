
#pragma once

#include "Simulation/Traffic/Maneuver/DeepDriveFourWayJunctionCalculator.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveTJunctionCalculator, Log, All);

class DeepDriveTJunctionCalculator	:	public DeepDriveFourWayJunctionCalculator
{
	enum Type
	{
		T_Junction,
		Left,
		Right
	};

public:

	DeepDriveTJunctionCalculator(const SDeepDriveRoadNetwork &roadNetwork);

	virtual void calculate(SDeepDriveManeuver &maneuver) override;

private:

	Type deduceJunctionType(TSortedJunctionEntries &sortedEntries);

};
