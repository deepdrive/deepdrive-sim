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
		float				Probability;

		float				Distance;
	};


	DeepDrivePredictedPath();
	~DeepDrivePredictedPath();

	void clear();
	void addPredictedPathPoint(float timeInFuture, const FVector &location, const FVector2D &direction, float velocity);
	void finalize();

	bool findIntersection(const DeepDrivePredictedPath &otherPath, SPredictedPathPoint *pathPoint, SPredictedPathPoint *otherPathPoint) const;

	bool advance(float DeltaTime);

	bool update(float predictionLength);

private :

	typedef TArray<SPredictedPathPoint> TPredictedPath;

	TPredictedPath                          m_PredictedPath;

	float									m_predictedDistance = 0.0f;

	const float								IntersectionThreshold = 100.0f;
};
