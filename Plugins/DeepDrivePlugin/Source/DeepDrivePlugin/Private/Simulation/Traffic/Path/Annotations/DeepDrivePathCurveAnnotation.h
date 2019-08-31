#pragma once

#include "CoreMinimal.h"
#include "Private/Simulation/Traffic/Path/DeepDrivePathDefines.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDrivePathCurveAnnotation, Log, All);

class DeepDrivePathCurveAnnotation
{

public:

	void annotate(TDeepDrivePathPoints &pathPoints);

};
