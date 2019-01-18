
#include "DeepDrivePluginPrivatePCH.h"
#include "Private/Simulation/RoadNetwork/DeepDriveRouteCalculator.h"

DeepDriveRouteCalculator::DeepDriveRouteCalculator(const SDeepDriveRoadNetwork &roadNetwork)
	:	m_RoadNetwork(roadNetwork)
{

}

SDeepDriveRouteData DeepDriveRouteCalculator::calculate(const FVector &start, const FVector &destination)
{
	SDeepDriveRouteData routeData;

	return routeData;
}
