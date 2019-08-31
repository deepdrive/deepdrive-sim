#pragma once

#include "CoreMinimal.h"
#include "Private/Simulation/Traffic/Path/DeepDrivePathDefines.h"

struct SDeepDriveRoadNetwork;

class DeepDrivePathSpeedAnnotation
{

public:

	DeepDrivePathSpeedAnnotation(const SDeepDriveRoadNetwork &roadNetwork, float speedDownRange, float speedUpRange);

	void annotate(TDeepDrivePathPoints &pathPoints);

private:

	const SDeepDriveRoadNetwork			&m_RoadNetwork;

	float								m_SpeedDownRange;
	float								m_SpeedUpRange;
};
