
#include "Simulation/Traffic/Path/Annotations/DeepDrivePathSpeedAnnotation.h"

DeepDrivePathSpeedAnnotation::DeepDrivePathSpeedAnnotation(const SDeepDriveRoadNetwork &roadNetwork, float speedDownRange, float speedUpRange)
	:	m_RoadNetwork(roadNetwork)
	,	m_SpeedDownRange(speedDownRange)
	,	m_SpeedUpRange(speedUpRange)
{

}

void DeepDrivePathSpeedAnnotation::annotate(TDeepDrivePathPoints &pathPoints)
{
	const int32 numPoints = pathPoints.Num();
	float speedLimit = DeepDriveRoadNetwork::SpeedLimitInTown;
	uint32 curSegmentId = 0;
	for (int32 i = 0; i < numPoints; ++i)
	{
		SDeepDrivePathPoint &pathPoint = pathPoints[i];
		if (pathPoint.SegmentId != curSegmentId)
		{
			curSegmentId = pathPoint.SegmentId;
			const SDeepDriveRoadSegment &segment = m_RoadNetwork.Segments[curSegmentId];
			const float speedLimitDelta = segment.SpeedLimit - speedLimit;

			if	(	i > 0
				// &&	segment.ConnectionShape != EDeepDriveConnectionShape::NO_CONNECTION
				&&	FMath::Abs(speedLimitDelta) > 1.0f
				)
			{
				if(speedLimitDelta < 0)
				{
					float distance = m_SpeedDownRange * (FMath::Abs(speedLimitDelta)) / 0.036f;
					const float speedLimitRampDown = speedLimitDelta / distance;

					FVector lastPoint = pathPoints[i].Location;
					for(int32 j = i - 1; j >= 0 && distance > 0.0f; --j)
					{
						const FVector &curPoint = pathPoints[j].Location;
						distance -= (curPoint - lastPoint).Size();
						if (distance > 0.0f)
						{
							pathPoints[j].SpeedLimit = speedLimit + distance * speedLimitRampDown;
							// pathPoints[j].SpeedLimit = segment.SpeedLimit;
							lastPoint = curPoint;
						}
						else
							break;
					}

				}
			}

			speedLimit = segment.SpeedLimit;
		}

		pathPoint.SpeedLimit = speedLimit;
	}
}
