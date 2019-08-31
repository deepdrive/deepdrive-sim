
#pragma once

#include "CoreMinimal.h"

struct SDeepDriveManeuver;
struct SDeepDriveJunctionEntry;
struct SDeepDriveRoadLink;
struct SDeepDriveJunction;
struct SDeepDriveRoadNetwork;

class DeepDriveJunctionCalculatorBase
{

public:

	DeepDriveJunctionCalculatorBase(const SDeepDriveRoadNetwork &roadNetwork);

	virtual ~DeepDriveJunctionCalculatorBase()
		{	}

	virtual void calculate(SDeepDriveManeuver &maneuver)
		{	}

protected:

	typedef TPair<float, const SDeepDriveJunctionEntry*>	TJunctionEntryItem;
	typedef TArray<TJunctionEntryItem>	TSortedJunctionEntries;

	float getRelativeDirection(const SDeepDriveRoadLink &fromLink, const SDeepDriveRoadLink &outLink);

	TSortedJunctionEntries collectJunctionEntries(const SDeepDriveJunction &junction, const SDeepDriveRoadLink &fromLink);


	const SDeepDriveRoadNetwork				&m_RoadNetwork;
};
