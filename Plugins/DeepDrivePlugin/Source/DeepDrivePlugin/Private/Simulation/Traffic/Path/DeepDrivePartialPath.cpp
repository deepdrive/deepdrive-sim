#include "Private/Simulation/Traffic/Path/DeepDrivePartialPath.h"
#include "Simulation/Traffic/Path/DeepDrivePathBuilder.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"
#include "Private/Simulation/Traffic/Path/DeepDrivePathDefines.h"
#include "Private/Simulation/Traffic/Path/Annotations/DeepDrivePathCurveAnnotation.h"
#include "Private/Simulation/Traffic/Path/Annotations/DeepDrivePathSpeedAnnotation.h"
#include "Private/Simulation/Traffic/Path/Annotations/DeepDrivePathDistanceAnnotation.h"
#include "Private/Simulation/Traffic/Path/Annotations/DeepDrivePathManeuverAnnotations.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBlackboard.h"

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
	pathBuilder.buildPath(baseSegments, m_Maneuvers);
	UE_LOG(LogDeepDrivePartialPath, Log, TEXT("Path built with %d points"), m_PathPoints.Num());
}

void DeepDrivePartialPath::trimStart(const FVector &startPos)
{
	const int32 numPoints = m_PathPoints.Num();
	float bestDistStart = TNumericLimits<float>::Max();
	int32 bestIndStart = -1;
	for (signed i = numPoints - 1; i >= 0; --i)
	{
		const float curDistStart = (startPos - m_PathPoints[i].Location).Size();
		if (curDistStart <= bestDistStart)
		{
			bestDistStart = curDistStart;
			bestIndStart = i;
		}
	}

	if (bestIndStart >= 0)
	{
		TDeepDrivePathPoints trimmedPath;
		for (int32 i = bestIndStart; i < numPoints; ++i)
			trimmedPath.Add(m_PathPoints[i]);

		m_PathPoints = trimmedPath;
		UE_LOG(LogDeepDrivePartialPath, Log, TEXT("Path trimmed to %d points from start"), m_PathPoints.Num());

		for(auto &man : m_Maneuvers)
		{
			if(man.EntryPointIndex >= bestIndStart)
				man.EntryPointIndex -= bestIndStart;
			if(man.ExitPointIndex >= bestIndStart)
				man.ExitPointIndex -= bestIndStart;
			if(man.DirectionIndicationBeginIndex >= bestIndStart)
				man.DirectionIndicationBeginIndex -= bestIndStart;
			if(man.DirectionIndicationEndIndex >= bestIndStart)
				man.DirectionIndicationEndIndex -= bestIndStart;
		}
	}
}

void DeepDrivePartialPath::trimEnd(const FVector &endPos)
{
	const int32 numPoints = m_PathPoints.Num();
	float bestDistEnd = TNumericLimits<float>::Max();
	int32 bestIndEnd = -1;
	for (signed i = numPoints - 1; i >= 0; --i)
	{
		const float curDistEnd = (endPos - m_PathPoints[i].Location).Size();
		if (curDistEnd <= bestDistEnd)
		{
			bestDistEnd = curDistEnd;
			bestIndEnd = i;
		}
	}

	if(bestIndEnd >= 0 && bestIndEnd < numPoints)
	{
		m_PathPoints.SetNum(bestIndEnd + 1);
		UE_LOG(LogDeepDrivePartialPath, Log, TEXT("Path trimmed to %d points from end"), m_PathPoints.Num());
	}

}

void DeepDrivePartialPath::annotate()
{
	DeepDrivePathManeuverAnnotations maneuverAnnotations;
	maneuverAnnotations.annotate(*this, m_PathPoints, m_Maneuvers);

	DeepDrivePathCurveAnnotation curveAnnotation;
	curveAnnotation.annotate(m_PathPoints);

	DeepDrivePathSpeedAnnotation speedAnnotation(m_RoadNetwork, 3.0f, 0.2f);
	speedAnnotation.annotate(m_PathPoints);

	DeepDrivePathDistanceAnnotation distanceAnnotation;
	distanceAnnotation.annotate(m_PathPoints, 0.0f);

	// int32 i = 0;
	// for(auto &pathPoint : m_PathPoints)
	// {
	// 	UE_LOG(LogDeepDrivePartialPath, Log, TEXT("%5d) spd %f rad %6.1f dist %f remDist %f"), i++, pathPoint.SpeedLimit, pathPoint.CurveRadius * 0.01f, pathPoint.Distance, pathPoint.RemainingDistance);
	// }
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

	steering = calcSteering(deltaSeconds);

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
		if	(	curManeuver->DirectionIndicationBeginIndex >= 0
			&&	m_curPathPointIndex >= curManeuver->DirectionIndicationBeginIndex
			)
		{
			switch(curManeuver->ManeuverType)
			{
				case EDeepDriveManeuverType::TURN_RIGHT:
					m_Agent.SetDirectionIndicatorState(EDeepDriveAgentDirectionIndicatorState::RIGHT);
					break;
				case EDeepDriveManeuverType::GO_ON_STRAIGHT:
					m_Agent.SetDirectionIndicatorState(EDeepDriveAgentDirectionIndicatorState::OFF);
					break;
				case EDeepDriveManeuverType::TURN_LEFT:
					m_Agent.SetDirectionIndicatorState(EDeepDriveAgentDirectionIndicatorState::LEFT);
					break;
			}
			curManeuver->DirectionIndicationBeginIndex = -1;
		}

		if	(	curManeuver->DirectionIndicationEndIndex >= 0
			&&	m_curPathPointIndex >= curManeuver->DirectionIndicationEndIndex
			)
		{
			m_Agent.SetDirectionIndicatorState(EDeepDriveAgentDirectionIndicatorState::UNKNOWN);
			curManeuver->DirectionIndicationEndIndex = -1;
		}

		curManeuver->BehaviorTree->execute(deltaSeconds, speed, m_curPathPointIndex);

		if	(	curManeuver->JunctionType == EDeepDriveJunctionType::DESTINATION_REACHED
			&&	m_hasReachedDestination == false
			)
		{
			m_hasReachedDestination = curManeuver->BehaviorTree->getBlackboard().getBooleanValue("DestinationReached", false);
		}

	}

	// UE_LOG(LogDeepDrivePartialPath, Log, TEXT("%5d) dist2ctr %f steering %f corr %f totE %f"), m_curPathPointIndex, dist2Ctr, steering, steeringCorrection, m_totalTrackError);
}

float DeepDrivePartialPath::getRouteLength() const
{
	return m_PathPoints.Num() > 0 ? m_PathPoints.Last().Distance : 0.0f; 
}

float DeepDrivePartialPath::getDistanceAlongRoute() const
{
	return		m_curPathPointIndex >= 0 && m_curPathPointIndex < m_PathPoints.Num()
			?	m_PathPoints.Last().Distance - m_PathPoints[m_curPathPointIndex].RemainingDistance
			:	0.0f;
}

float DeepDrivePartialPath::getDistanceToCenterOfTrack() const
{
	float dist2Ctr = 0.0f;
	if(m_curPathPointIndex >= 0 && m_curPathPointIndex < m_PathPoints.Num())
	{

	}
	return dist2Ctr;
}


float DeepDrivePartialPath::calcSteering(float dT)
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

int32 DeepDrivePartialPath::windForward(int32 fromIndex, float distanceCM, float *remainingDistance) const
{
	const int32_t numPoints = m_PathPoints.Num() - 2;
	while(fromIndex < numPoints && distanceCM > 0.0f)
	{
		const float curLength = (m_PathPoints[fromIndex - 1].Location - m_PathPoints[fromIndex].Location).Size();
		++fromIndex;
		distanceCM -= curLength;
	}

	if(remainingDistance)
		*remainingDistance = distanceCM;

	return fromIndex;
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
