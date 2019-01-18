
#pragma once

#include "Public/Simulation/RoadNetwork/DeepDriveRoadNetwork.h"

class DeepDriveRouteCalculator
{
public:

    DeepDriveRouteCalculator(const SDeepDriveRoadNetwork &roadNetwork);

	SDeepDriveRouteData calculate(const FVector &start, const FVector &destination);


private:

    const SDeepDriveRoadNetwork         &m_RoadNetwork;

};
