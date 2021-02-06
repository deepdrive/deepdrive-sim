#pragma once

#include "CoreMinimal.h"
#include "Simulation/Traffic/Path/DeepDrivePathDefines.h"

struct SDeepDriveRoadNetwork;

class DeepDrivePathDistanceAnnotation
{

public:

	void annotate(TDeepDrivePathPoints &pathPoints, float startDistance);

private:

};
