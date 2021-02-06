
#include "Simulation/Traffic/Path/Annotations/DeepDrivePathCurveAnnotation.h"
#include "Simulation/Misc/MedianLowPassFilter.h"

DEFINE_LOG_CATEGORY(LogDeepDrivePathCurveAnnotation);

void DeepDrivePathCurveAnnotation::annotate(TDeepDrivePathPoints &pathPoints)
{
	for(int32 i = 1; i < pathPoints.Num() - 1; ++i)
	{
		FVector2D p0 = FVector2D(pathPoints[i - 1].Location);
		FVector2D p1 = FVector2D(pathPoints[i].Location);
		FVector2D p2 = FVector2D(pathPoints[i + 1].Location);

		const float A = p0.X * (p1.Y - p2.Y) - p0.Y * (p1.X - p2.X) + p1.X * p2.Y - p2.X * p1.Y;
		const float B = (p0.X * p0.X + p0.Y * p0.Y) * (p2.Y - p1.Y) + (p1.X * p1.X + p1.Y * p1.Y) * (p0.Y - p2.Y) + (p2.X * p2.X + p2.Y * p2.Y) * (p1.Y - p0.Y);
		const float C = (p0.X * p0.X + p0.Y * p0.Y) * (p1.X - p2.X) + (p1.X * p1.X + p1.Y * p1.Y) * (p2.X - p0.X) + (p2.X * p2.X + p2.Y * p2.Y) * (p0.X - p1.X);
		const float D = (p0.X * p0.X + p0.Y * p0.Y) * (p2.X * p1.Y - p1.X * p2.Y) + (p1.X * p1.X + p1.Y * p1.Y) * (p0.X * p2.Y - p2.X * p0.Y) + (p2.X * p2.X + p2.Y * p2.Y) * (p1.X * p0.Y - p0.X * p1.Y);

		FVector2D dir(p1 - p0);
		FVector2D s(p2 - p1);
		dir.Normalize();
		s.Normalize();
		FVector2D nrm(dir.Y, -dir.X);

		float radius = 0.0f;
		if (FMath::Abs(A) > 0.0001f)
		{
			radius = FMath::Sqrt( (B * B + C * C - 4.0f * A * D) / (4.0f * A * A) );
			// if(radius > 50000.0f)
			//	radius = 0.0f;
			// else
				radius *= FVector2D::DotProduct(nrm, s) >= 0.0f ? -1.0f : 1.0f;
		}

		pathPoints[i].CurveRadius = radius;
		pathPoints[i].Normal = nrm;
		const float curveAngle = radius != 0.0f ? FMath::RadiansToDegrees( FMath::Asin( 300.0f / radius)) : 0.0f;

		FVector2D d0(p1 - p0);
		FVector2D d1(p2 - p1);
		d0.Normalize();
		d1.Normalize();

		pathPoints[i].CurveAngle = FMath::RadiansToDegrees(FMath::Acos(FVector2D::DotProduct(d0, d1)));
		pathPoints[i].Heading = FVector(d0, 0.0f).HeadingAngle();

		// UE_LOG(LogDeepDrivePathCurveAnnotation, Log, TEXT("%5d) ang %f  %f hdg %f"), i, pathPoints[i].CurveAngle, curveAngle, pathPoints[i].Heading);
		// UE_LOG(LogDeepDrivePartialPath, Log, TEXT("  %f   %f => %f"), A, radius,  curveAngle);
	}
	pathPoints[0].CurveRadius = pathPoints[1].CurveRadius;
	pathPoints[0].Normal = pathPoints[1].Normal;
	pathPoints[pathPoints.Num() - 1].CurveRadius = pathPoints[pathPoints.Num() - 2].CurveRadius;
	pathPoints[pathPoints.Num() - 1].Normal = pathPoints[pathPoints.Num() - 2].Normal;

	TMedianLowPassFilter<3> filter;
	for(auto &pathPoint : pathPoints)
	{
		pathPoint.CurveRadius = filter.add(pathPoint.CurveRadius);
	}
}
