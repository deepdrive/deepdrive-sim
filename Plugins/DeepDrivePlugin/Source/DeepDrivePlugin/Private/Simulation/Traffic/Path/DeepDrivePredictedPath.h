#pragma once

#include "CoreMinimal.h"

class DeepDrivePredictedPath
{
private:

	struct SPredictedPathPoint
	{
		float				TimeInFuture;
		FVector				Location;
		float				Velocity;

		float				Probability;
		float				Distance;
	};

public:

	struct SPathIntersectionPoint
	{
		float				TimeInFuture;
		FVector				Location;
		float				Velocity;
	};

	DeepDrivePredictedPath();
	~DeepDrivePredictedPath();

	void clear();
	void addPredictedPathPoint(float timeInFuture, const FVector &location, float velocity, float probability);
	void finalize();

	void advance(float deltaSeconds, const FVector &curLocation);
	bool isReliable(float distance) const;

	bool findIntersection(const DeepDrivePredictedPath &otherPath, SPathIntersectionPoint *pathPoint, SPathIntersectionPoint *otherPathPoint) const;

private :

	typedef TArray<SPredictedPathPoint> TPredictedPath;

	TPredictedPath                          m_PredictedPath;

	float									m_BaseTime = 0.0f;
	float									m_predictedDistance = 0.0f;

	const float								IntersectionThreshold = 100.0f;
};
