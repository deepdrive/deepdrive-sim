
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

		FVector direction = pathPoint.Location - lastLocation;

		const float curDistance = direction.Size();
		coveredDistance += curDistance;
		pathPoint.Distance = coveredDistance;
		direction.Normalize();
		pathPoints[i - 1].Direction = FVector2D(direction);

		lastLocation = pathPoint.Location;
	}

	for(int32 i = pathPoints.Num() - 1; i >= 0; --i)
		pathPoints[i].RemainingDistance = coveredDistance - pathPoints[i].Distance;
}
