#pragma once

#include "CoreMinimal.h"
#include "Private/Simulation/Traffic/Path/DeepDrivePathDefines.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDrivePathManeuverAnnotations, Log, All);

struct SDeepDriveRoadNetwork;
class DeepDrivePartialPath;

class DeepDrivePathManeuverAnnotations
{

public:

	void annotate(DeepDrivePartialPath &path, TDeepDrivePathPoints &pathPoints, TDeepDriveManeuvers &maneuvers);

};
