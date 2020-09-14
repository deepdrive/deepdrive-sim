
#include "Private/Simulation/Traffic/Path/DeepDrivePredictedPath.h"

#include <limits>

DeepDrivePredictedPath::DeepDrivePredictedPath()
{
}

DeepDrivePredictedPath::~DeepDrivePredictedPath()
{
}

void DeepDrivePredictedPath::clear()
{
	m_PredictedPath.Empty();
}

void DeepDrivePredictedPath::addPredictedPathPoint(float timeInFuture, const FVector &location, const FVector2D &direction, float velocity)
{
	SPredictedPathPoint ppp;
	ppp.TimeInFuture = timeInFuture;
	ppp.Location = location;
	ppp.Direction = direction;
	ppp.Velocity = velocity;
	m_PredictedPath.Add(ppp);
}

void DeepDrivePredictedPath::finalize()
{
	//	 sort by time
}

bool DeepDrivePredictedPath::findIntersection(const DeepDrivePredictedPath &otherPath, SPredictedPathPoint *pathPoint, SPredictedPathPoint *otherPathPoint) const
{
	bool intersectionFound = false;

	float closestDist = std::numeric_limits<float>::max();
	int32 ind0 = -1;
	int32 ind1 = -1;

	for(signed i = 0; i < m_PredictedPath.Num(); ++i)
	{
		const auto &p0 = m_PredictedPath[i];
		for(signed j = 0; j < otherPath.m_PredictedPath.Num(); ++j)
		{
			const auto &p1 = otherPath.m_PredictedPath[j];
			const float curDist = (p0.Location - p1.Location).Size();
			if(curDist < closestDist)
			{
				closestDist = curDist;
				ind0 = i;
				ind1 = j;
			}
		}
	}

	if(closestDist < IntersectionThreshold)
	{
		if(pathPoint)
		{
			*pathPoint = m_PredictedPath[ind0];
		}
		if(otherPathPoint)
		{
			*otherPathPoint = otherPath.m_PredictedPath[ind0];
		}
		intersectionFound = true;
	}

	return intersectionFound;
}
