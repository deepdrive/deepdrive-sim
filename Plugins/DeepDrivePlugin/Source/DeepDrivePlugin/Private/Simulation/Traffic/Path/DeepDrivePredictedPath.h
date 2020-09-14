#pragma once

#include "CoreMinimal.h"

class DeepDrivePredictedPath
{
public:

	struct SPredictedPathPoint
	{
		float				TimeInFuture;

		FVector				Location;
		FVector2D			Direction;
		float				Velocity;
	};


	DeepDrivePredictedPath();
	~DeepDrivePredictedPath();

	void clear();
	void addPredictedPathPoint(float timeInFuture, const FVector &location, const FVector2D &direction, float velocity);
	void finalize();

	bool findIntersection(const DeepDrivePredictedPath &otherPath, SPredictedPathPoint *pathPoint, SPredictedPathPoint *otherPathPoint) const;

	bool moveIntoFuture(float DeltaTime);

private:

	typedef TArray<SPredictedPathPoint> TPredictedPath;

	TPredictedPath                          m_PredictedPath;

	const float								IntersectionThreshold = 100.0f;
};
