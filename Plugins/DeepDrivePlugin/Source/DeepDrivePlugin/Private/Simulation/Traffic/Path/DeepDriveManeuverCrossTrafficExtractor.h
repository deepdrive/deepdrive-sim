
#pragma once

#include "CoreMinimal.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveManeuverCrossTrafficExtractor, Log, All);

struct SDeepDriveRoadNetwork;
struct SDeepDriveManeuver;
struct SDeepDriveRoadLink;

class DeepDriveManeuverCrossTrafficExtractor
{
public:

	DeepDriveManeuverCrossTrafficExtractor(const SDeepDriveRoadNetwork &roadNetwork);

	void extract(SDeepDriveManeuver &maneuver);

private:

	void extractFromCrossJunction(SDeepDriveManeuver &maneuver);

	float getRelativeDirection(const SDeepDriveRoadLink &fromLink, const SDeepDriveRoadLink &outLink);

	const SDeepDriveRoadNetwork				&m_RoadNetwork;

};
