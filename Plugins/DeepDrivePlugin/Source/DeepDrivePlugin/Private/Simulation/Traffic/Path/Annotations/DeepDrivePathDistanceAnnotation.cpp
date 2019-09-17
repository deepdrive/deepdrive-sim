
#include "Private/Simulation/Traffic/Path/Annotations/DeepDrivePathDistanceAnnotation.h"


void DeepDrivePathDistanceAnnotation::annotate(TDeepDrivePathPoints &pathPoints, float startDistance)
{
	const int32 numPoints = pathPoints.Num();
	float coveredDistance = startDistance;
	FVector lastLocation = pathPoints[0].Location;
	pathPoints[0].Distance = coveredDistance;
	for (int32 i = 1; i < numPoints; ++i)
	{
		SDeepDrivePathPoint &pathPoint = pathPoints[i];

		const float curDistance = (pathPoint.Location - lastLocation).Size();
		coveredDistance += curDistance;
		pathPoint.Distance = coveredDistance;

		lastLocation = pathPoint.Location;
	}

	for(int32 i = pathPoints.Num() - 1; i >= 0; --i)
		pathPoints[i].RemainingDistance = coveredDistance - pathPoints[i].Distance;
}
