
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

void DeepDrivePredictedPath::addPredictedPathPoint(float timeInFuture, const FVector &location, float velocity, float probability)
{
	SPredictedPathPoint ppp;
	ppp.TimeInFuture = timeInFuture;
	ppp.Location = location;
	ppp.Velocity = velocity;
	ppp.Probability = probability;
	m_PredictedPath.Add(ppp);
}

void DeepDrivePredictedPath::finalize()
{
	const int32 numPoints = m_PredictedPath.Num();
	if(numPoints > 0)
	{
		float distance = 0.0f;
		m_PredictedPath[0].Distance = distance;
		for(signed i = 1; i < numPoints; ++i)
		{
			const float curDistance = (m_PredictedPath[i].Location - m_PredictedPath[i - 1].Location).Size();
			distance += curDistance;
			m_PredictedPath[i].Distance = distance;
		}
	}
}

void DeepDrivePredictedPath::advance(float deltaSeconds, const FVector &curLocation)
{

}

bool DeepDrivePredictedPath::isReliable(float distance) const
{
	return false;
}

bool DeepDrivePredictedPath::findIntersection(const DeepDrivePredictedPath &otherPath, SPathIntersectionPoint *pathPoint, SPathIntersectionPoint *otherPathPoint) const
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
			pathPoint->TimeInFuture = m_PredictedPath[ind0].TimeInFuture - m_BaseTime;
			pathPoint->Location = m_PredictedPath[ind0].Location;
			pathPoint->Velocity = m_PredictedPath[ind0].Velocity;
		}
		if(otherPathPoint)
		{
			otherPathPoint->TimeInFuture = otherPath.m_PredictedPath[ind1].TimeInFuture - otherPath.m_BaseTime;
			otherPathPoint->Location = otherPath.m_PredictedPath[ind1].Location;
			otherPathPoint->Velocity = otherPath.m_PredictedPath[ind1].Velocity;
		}
		intersectionFound = true;
	}

	return intersectionFound;
}
