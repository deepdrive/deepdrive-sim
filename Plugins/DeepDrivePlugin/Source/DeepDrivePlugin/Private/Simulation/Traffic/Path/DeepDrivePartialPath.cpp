#include "Private/Simulation/Traffic/Path/DeepDrivePartialPath.h"
#include "Simulation/Traffic/Path/DeepDrivePathBuilder.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"
#include "Private/Simulation/Traffic/Path/DeepDrivePathDefines.h"
#include "Private/Simulation/Traffic/Path/Annotations/DeepDrivePathCurveAnnotation.h"
#include "Private/Simulation/Traffic/Path/Annotations/DeepDrivePathSpeedAnnotation.h"
#include "Private/Simulation/Traffic/Path/Annotations/DeepDrivePathDistanceAnnotation.h"
#include "Private/Simulation/Traffic/Path/Annotations/DeepDrivePathManeuverAnnotations.h"

#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBehaviorTree.h"

#include "Runtime/Engine/Classes/Kismet/KismetMathLibrary.h"

DEFINE_LOG_CATEGORY(LogDeepDrivePartialPath);

DeepDrivePartialPath::DeepDrivePartialPath(ADeepDriveAgent &agent, const SDeepDriveRoadNetwork &roadNetwork, UBezierCurveComponent &bezierCmp, const SDeepDrivePathConfiguration &pathCfg)
	:	m_Agent(agent)
	,	m_RoadNetwork(roadNetwork)
	,	m_BezierCurve(bezierCmp)
	,	m_PathConfiguration(pathCfg)
	,	m_SteeringPIDCtrl(pathCfg.PIDSteering.X, pathCfg.PIDSteering.Y, pathCfg.PIDSteering.Z)
{

}

void DeepDrivePartialPath::setup(const TArray<SDeepDriveBasePathSegment> &baseSegments)
{
	DeepDrivePathBuilder pathBuilder(m_RoadNetwork, m_PathPoints, m_BezierCurve);
	pathBuilder.buildPath(baseSegments);

	for (const SDeepDriveBasePathSegment &baseSegment : baseSegments)
		if (baseSegment.Maneuver.FromLinkId && baseSegment.Maneuver.ToLinkId && baseSegment.Maneuver.BehaviorTree)
			m_Maneuvers.Add(baseSegment.Maneuver);

	{
		DeepDrivePathCurveAnnotation curveAnnotation;
		curveAnnotation.annotate(m_PathPoints);

		DeepDrivePathSpeedAnnotation speedAnnotation(m_RoadNetwork, 3.0f, 0.2f);
		speedAnnotation.annotate(m_PathPoints);

		DeepDrivePathDistanceAnnotation distanceAnnotation;
		distanceAnnotation.annotate(m_PathPoints, 0.0f);

		DeepDrivePathManeuverAnnotations maneuverAnnotations;
		maneuverAnnotations.annotate(*this, m_PathPoints, m_Maneuvers);


		// int32 i = 0;
		// for(auto &pathPoint : m_PathPoints)
		// {
		// 	UE_LOG(LogDeepDrivePartialPath, Log, TEXT("%5d) spd %f rad %6.1f dist %f remDist %f"), i++, pathPoint.SpeedLimit, pathPoint.CurveRadius * 0.01f, pathPoint.Distance, pathPoint.RemainingDistance);
		// }
	}
}

bool DeepDrivePartialPath::update()
{
	bool upToDate = true;

	m_curAgentLocation = m_Agent.GetActorLocation();
	findClosestPathPoint(m_curAgentLocation);
	
	return true;
}

void DeepDrivePartialPath::advance(float deltaSeconds, float &speed, float &steering, float &brake)
{
	const SDeepDrivePathPoint &curPathPoint = m_PathPoints[m_curPathPointIndex];
	speed = FMath::Min(curPathPoint.SpeedLimit, speed);
	speed *= 1.0f - SmootherStep(0.50f, 1.0f, FMath::Abs(steering)) * 0.5f;

	brake = speed > 0.0f ? 0.0f : 1.0f;

	//steering = calcSteering(deltaSeconds);
	//steering = calcSteering_Radius(deltaSeconds);
	// steering = calcSteering_Angle(deltaSeconds);
	// steering = calcSteering_Heading(deltaSeconds);
	steering = calcSteering_Classic(deltaSeconds);

	SDeepDriveManeuver *curManeuver = 0;
	for(auto &m : m_Maneuvers)
	{
		if	(	m_curPathPointIndex >= m.EntryPointIndex
			&&	m_curPathPointIndex <= m.ExitPointIndex
			)
		{
			curManeuver = &m;
			// UE_LOG(LogDeepDrivePartialPath, Log, TEXT("Found maneuver at %d from %d to %d"), m_curPathPointIndex, m.EntryPointIndex, m.ExitPointIndex);
			break;
		}
	}

	if	(	curManeuver
		&&	curManeuver->BehaviorTree
		)
	{
		curManeuver->BehaviorTree->execute(deltaSeconds, speed, m_curPathPointIndex);
	}

	// UE_LOG(LogDeepDrivePartialPath, Log, TEXT("%5d) dist2ctr %f steering %f corr %f totE %f"), m_curPathPointIndex, dist2Ctr, steering, steeringCorrection, m_totalTrackError);
}

bool DeepDrivePartialPath::isCloseToEnd(float distanceFromEnd) const
{
    return m_PathPoints[m_curPathPointIndex].RemainingDistance <= distanceFromEnd;
}

float DeepDrivePartialPath::calcSteering(float dT)
{
	const SDeepDrivePathPoint &curPathPoint = m_PathPoints[m_curPathPointIndex];
	const SDeepDrivePathPoint &nextPathPoint = m_PathPoints[m_curPathPointIndex < (m_PathPoints.Num() - 1) ? m_curPathPointIndex + 1 : m_curPathPointIndex];

	float steering = 0.0f;

	FVector2D loc(m_curAgentLocation);
	FVector2D dir2Loc(loc - FVector2D(curPathPoint.Location));

	float dist2Ctr = FVector2D::DotProduct(dir2Loc, FMath::Lerp(curPathPoint.Normal, nextPathPoint.Normal, m_curPathSegmentT));

	float steeringCorrection = FMath::SmoothStep(0.0f, 75.0f, FMath::Abs(dist2Ctr)) * 0.2f * FMath::Sign(dist2Ctr);

	steeringCorrection = SmootherStep(m_PathConfiguration.PIDSteering.X, m_PathConfiguration.PIDSteering.Y, FMath::Abs(dist2Ctr)) * m_PathConfiguration.PIDSteering.Z * FMath::Sign(dist2Ctr);

	steeringCorrection = FMath::Clamp(m_SteeringPIDCtrl.advance(dT, dist2Ctr), -10.5f, 10.5f);

	steering = FMath::Clamp(steeringCorrection, -1.0f, 1.0f);

	//steering = m_SteeringSmoother.add(steering);

	m_totalTrackError += (m_curPathPointIndex < 50 ? 0.0f : 1.0f) * FMath::Abs(dist2Ctr);
	UE_LOG(LogDeepDrivePartialPath, Log, TEXT("%5d) curE %f  dE %f sumE %f steering %f"), m_curPathPointIndex, m_SteeringPIDCtrl.m_curE, m_SteeringPIDCtrl.m_curDE, m_SteeringPIDCtrl.m_curSumE, steering);

	return steering;
}

float DeepDrivePartialPath::calcSteering_Radius(float dT)
{
	const SDeepDrivePathPoint &curPathPoint = m_PathPoints[m_curPathPointIndex];
	const SDeepDrivePathPoint &nextPathPoint = m_PathPoints[m_curPathPointIndex < (m_PathPoints.Num() - 1) ? m_curPathPointIndex + 1 : m_curPathPointIndex];
	const float maxSteeringAngle = 50.0f;
	const float WheelBase = m_Agent.getWheelBase();

	const float curveRadius = curPathPoint.CurveRadius;
	const float steeringAngle = curveRadius != 0.0f ? FMath::RadiansToDegrees(FMath::Asin(WheelBase / curveRadius)) : 0.0f;

	float steering = steeringAngle / maxSteeringAngle;

	FVector2D loc(m_curAgentLocation);
	FVector2D dir2Loc(loc - FVector2D(curPathPoint.Location));

	float dist2Ctr = FVector2D::DotProduct(dir2Loc, FMath::Lerp(curPathPoint.Normal, nextPathPoint.Normal, m_curPathSegmentT));

	float steeringCorrection = FMath::SmoothStep(0.0f, 75.0f, FMath::Abs(dist2Ctr)) * 0.2f * FMath::Sign(dist2Ctr);

	steeringCorrection = SmootherStep(m_PathConfiguration.PIDSteering.X, m_PathConfiguration.PIDSteering.Y, FMath::Abs(dist2Ctr)) * m_PathConfiguration.PIDSteering.Z * FMath::Sign(dist2Ctr);

	//steering += steeringCorrection;

	//steering = m_SteeringSmoother.add(steering);

	m_totalTrackError += (m_curPathPointIndex < 50 ? 0.0f : 1.0f) * FMath::Abs(dist2Ctr);
	UE_LOG(LogDeepDrivePartialPath, Log, TEXT("%5d) curveRad %f  steering %f"), m_curPathPointIndex, curveRadius, steering);

	//UE_LOG(LogDeepDrivePartialPath, Log, TEXT("%5d) curE %f  dE %f sumE %f steering %f"), m_curPathPointIndex, m_SteeringPIDCtrl.m_curE, m_SteeringPIDCtrl.m_curDE, m_SteeringPIDCtrl.m_curSumE, steering);

	return steering;
}

float DeepDrivePartialPath::calcSteering_Angle(float dT)
{
	const SDeepDrivePathPoint &curPathPoint = m_PathPoints[m_curPathPointIndex];
	const SDeepDrivePathPoint &nextPathPoint = m_PathPoints[m_curPathPointIndex < (m_PathPoints.Num() - 1) ? m_curPathPointIndex + 1 : m_curPathPointIndex];

	float curAng = curPathPoint.CurveAngle + nextPathPoint.CurveAngle + m_PathPoints[m_curPathPointIndex + 2].CurveAngle;

	float steering = 2.0f * curAng / 50.0f;
	steering = FMath::Clamp(steering, -1.0f, 1.0f);

	UE_LOG(LogDeepDrivePartialPath, Log, TEXT("%5d) curveAng %f  steering %f"), m_curPathPointIndex, curAng, steering);

	return steering;
}

float DeepDrivePartialPath::calcSteering_Heading(float dT)
{
	const SDeepDrivePathPoint &curPathPoint = m_PathPoints[m_curPathPointIndex];
	const SDeepDrivePathPoint &nextPathPoint = m_PathPoints[m_curPathPointIndex < (m_PathPoints.Num() - 1) ? m_curPathPointIndex + 1 : m_curPathPointIndex];

	float desiredHdg = FMath::Lerp(curPathPoint.Heading, nextPathPoint.Heading, m_curPathSegmentT);
	float curHdg = m_Agent.GetActorForwardVector().HeadingAngle();
	float delta = desiredHdg - curHdg;

	float steering = m_SteeringPIDCtrl.advance(dT, delta) * dT;

	const FVector &posAhead = m_PathPoints[m_curPathPointIndex + 3].Location;
	FVector desiredForward = posAhead - m_curAgentLocation;
	desiredForward.Z = 0.0f;
	desiredForward.Normalize();
	desiredHdg = FMath::RadiansToDegrees(desiredForward.HeadingAngle());

	FVector curForward = m_Agent.GetActorForwardVector();
	curForward.Z = 0.0f;
	curHdg = FMath::RadiansToDegrees(curForward.HeadingAngle());

	delta = desiredHdg - curHdg;
	if (delta > 180.0f)
	{
		delta -= 360.0f;
	}

	if (delta < -180.0f)
	{
		delta += 360.0f;
	}

	steering = m_SteeringPIDCtrl.advance(dT, delta) * dT;
	steering = FMath::Clamp(steering, -1.0f, 1.0f);

	steering = m_SteeringSmoother.add(steering);

	steering = FMath::Clamp(steering, -1.0f, 1.0f);

	UE_LOG(LogDeepDrivePartialPath, Log, TEXT("%5d) desiredHdg %f curHdg %f delta %f steering %f"), m_curPathPointIndex, desiredHdg, curHdg, delta, steering);

	return steering;
}

float DeepDrivePartialPath::calcSteering_Classic(float dT)
{
	const int32 ind0 = FMath::Min(m_curPathPointIndex + 4, m_PathPoints.Num() - 1);
	const int32 ind1 = FMath::Min(m_curPathPointIndex + 5, m_PathPoints.Num() - 1);
	const SDeepDrivePathPoint &pathPoint0 = m_PathPoints[ind0];
	const SDeepDrivePathPoint &pathPoint1 = m_PathPoints[ind1];
	const FVector &posAhead = FMath::Lerp(pathPoint0.Location, pathPoint1.Location, m_curPathSegmentT);

	FVector desiredForward = posAhead - m_curAgentLocation;
	desiredForward.Normalize();

	float curYaw = m_Agent.GetActorRotation().Yaw;
	float desiredYaw = FMath::RadiansToDegrees(FMath::Atan2(desiredForward.Y, desiredForward.X));
	float delta = desiredYaw - curYaw;
	if (delta > 180.0f)
		delta -= 360.0f;
	else if (delta < -180.0f)
		delta += 360.0f;

	float steering = m_SteeringPIDCtrl.advance(dT, delta);
	steering = FMath::Clamp(steering, -1.0f, 1.0f);

	//steering = m_SteeringSmoother.add(steering);

	// UE_LOG(LogDeepDrivePartialPath, Log, TEXT("%5d) curE %f  dE %f sumE %f steering %f"), m_curPathPointIndex, m_SteeringPIDCtrl.m_curE, m_SteeringPIDCtrl.m_curDE, m_SteeringPIDCtrl.m_curSumE, steering);

	return steering;
}

void DeepDrivePartialPath::findClosestPathPoint(const FVector &location)
{
#if 0
	float bestDist = TNumericLimits<float>::Max();
	int32 index = -1;
	for (signed i = m_PathPoints.Num() - 1; i >= 0; --i)
	{
		const float curDist = (location - m_PathPoints[i].Location).Size();
		if (curDist <= bestDist)
		{
			bestDist = curDist;
			index = i;
		}
	}
	m_curPathPointIndex = index;

#else

	int32 index = -1;
	float bestDist = TNumericLimits<float>::Max();
	for (int32 i = 0; i < m_PathPoints.Num() - 2; ++i)
	{
		const FVector curPoint = UKismetMathLibrary::FindClosestPointOnSegment(location, m_PathPoints[i].Location, m_PathPoints[i + 1].Location);
		const float curDist = (location - curPoint).Size();
		if (curDist <= bestDist)
		{
			bestDist = curDist;
			index = i;
		}
	}

	if(index >=0)
	{
		m_curPathPointIndex = index;

		FVector a = m_PathPoints[index + 1].Location - m_PathPoints[index].Location;
		FVector b = location - m_PathPoints[index].Location;
		const float al = a.Size();
		const float bl = b.Size();
		a.Normalize();
		b.Normalize();
		const float cosA = FVector::DotProduct(a, b);
		m_curPathSegmentT = FMath::Abs(cosA) > 0.0001 ? (bl / cosA) / al : 0.0f;
	}

#endif
}

int32 DeepDrivePartialPath::findClosestPathPoint(const FVector &location, float *t)
{
	int32 index = -1;
	float bestDist = TNumericLimits<float>::Max();
	for (int32 i = 0; i < m_PathPoints.Num() - 2; ++i)
	{
		const FVector curPoint = UKismetMathLibrary::FindClosestPointOnSegment(location, m_PathPoints[i].Location, m_PathPoints[i + 1].Location);
		const float curDist = (location - curPoint).Size();
		if (curDist <= bestDist)
		{
			bestDist = curDist;
			index = i;
		}
	}

	if(t != 0 && index >=0 && index)
	{
		m_curPathPointIndex = index;

		FVector a = m_PathPoints[index + 1].Location - m_PathPoints[index].Location;
		FVector b = location - m_PathPoints[index].Location;
		const float al = a.Size();
		const float bl = b.Size();
		a.Normalize();
		b.Normalize();
		const float cosA = FVector::DotProduct(a, b);
		*t = FMath::Abs(cosA) > 0.0001 ? (bl / cosA) / al : 0.0f;
	}

	return index;
}

int32 DeepDrivePartialPath::rewind(int32 fromIndex, float distanceCM, float *remainingDistance) const
{
	while(fromIndex > 0 && distanceCM > 0.0f)
	{
		const float curLength = (m_PathPoints[fromIndex].Location - m_PathPoints[fromIndex - 1].Location).Size();
		--fromIndex;
		distanceCM -= curLength;
	}

	if(remainingDistance)
		*remainingDistance = distanceCM;

	return fromIndex;
}

float DeepDrivePartialPath::getDistance(int32 fromIndex, int32 toIndex)
{
	return		toIndex >= fromIndex && fromIndex >= 0
			?	FMath::Max(static_cast<float>(toIndex - fromIndex), 0.0f) * m_StepSize
			:	-1.0f;
}

void DeepDrivePartialPath::showPath(UWorld *world)
{
	FColor col = FColor::Green;
	const uint8 prio = 40;

	DrawDebugPoint(world, m_PathPoints[0].Location, 10.0f, col, false, 0.0f, prio);
	for(signed i = 1; i < m_PathPoints.Num(); ++i)
	{
		DrawDebugPoint(world, m_PathPoints[i].Location, 10.0f, col, false, 0.0f, prio);
		DrawDebugLine(world, m_PathPoints[i - 1].Location, m_PathPoints[i].Location, col, false, 0.0f, prio, 4.0f);
	}
}

float DeepDrivePartialPath::SmootherStep(float edge0, float edge1, float val)
{
	float x = FMath::Clamp((val - edge0) / (edge1 - edge0), 0.0f, 1.0f);
	return x * x * x * (x * (x * 6.0f - 15.0f) + 10.0f);
}
