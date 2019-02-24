

#include "DeepDrivePluginPrivatePCH.h"
#include "BezierCurveComponent.h"

// Sets default values for this component's properties
UBezierCurveComponent::UBezierCurveComponent()
{
	// Set this component to be initialized when the game starts, and to be ticked every frame.  You can turn these features
	// off to improve performance if you don't need them.
	PrimaryComponentTick.bCanEverTick = true;

	// ...
}


// Called when the game starts
void UBezierCurveComponent::BeginPlay()
{
	Super::BeginPlay();

	// ...
	
}


// Called every frame
void UBezierCurveComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	// ...
}

void UBezierCurveComponent::ClearControlPoints()
{
	m_ControlPoints.Empty();
}

void UBezierCurveComponent::AddControlPoint(const FVector &ControlPoint)
{
	m_ControlPoints.Add(ControlPoint);
}

void UBezierCurveComponent::UpdateControlPoint(int32 Index, const FVector &ControlPoint)
{
	if(Index >= 0 && Index < m_ControlPoints.Num())
	{
		m_ControlPoints[Index] = ControlPoint;
	}
}

FVector UBezierCurveComponent::Evaluate(float T)
{
	T = FMath::Clamp(T, 0.0f, 1.0f);
	if(m_ControlPoints.Num() == 3)
	{
		return evaluateQuadraticSpline(T);
	}
	else if(m_ControlPoints.Num() == 4)
	{
		return evaluateCubicSpline(T);
	}
	else if(m_ControlPoints.Num() > 4)
	{
		return splitDeCasteljau(m_ControlPoints, T);
	}
	return FVector();
}

FVector UBezierCurveComponent::evaluateQuadraticSpline(float t)
{
	const float oneMinusT = (1.0f - t);
	const FVector &p0 = m_ControlPoints[0];
	const FVector &p1 = m_ControlPoints[1];
	const FVector &p2 = m_ControlPoints[2];
	FVector pos = oneMinusT * oneMinusT * p0 + 2.0f * oneMinusT * t * p1 + t * t *p2;
	return pos;
}

FVector UBezierCurveComponent::evaluateCubicSpline(float t)
{
	const float oneMinusT = (1.0f - t);
	const FVector &p0 = m_ControlPoints[0];
	const FVector &p1 = m_ControlPoints[1];
	const FVector &p2 = m_ControlPoints[2];
	const FVector &p3 = m_ControlPoints[3];
	FVector pos = oneMinusT * oneMinusT * oneMinusT * p0 + (3.0f * oneMinusT * oneMinusT * t) * p1 + (3.0f * oneMinusT * t * t) * p2 + t * t * t * p3;
	return pos;
}

FVector UBezierCurveComponent::splitDeCasteljau(const TArray<FVector> &points, float t)
{
	const float t1 = 1.0f - t;
	if(points.Num() == 2)
	{
		return points[0] * t1 + points[1] * t;
	}

	TArray<FVector> newPoints;
	for(int32 i = 0; i < points.Num() - 1; ++i)
	{
		newPoints.Add(points[i] * t1 + points[i + 1] * t);
	}
	return splitDeCasteljau(newPoints, t);
}
