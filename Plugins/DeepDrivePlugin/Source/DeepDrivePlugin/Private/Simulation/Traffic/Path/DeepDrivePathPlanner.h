#pragma once

#include "CoreMinimal.h"

struct SDeepDriveRoadNetwork;
class DeepDriveBasePath;
class ADeepDriveAgent;
class UBezierCurveComponent;
struct SDeepDrivePathConfiguration;
struct SDeepDriveRoute;

class DeepDrivePathPlanner
{

public:

	DeepDrivePathPlanner(ADeepDriveAgent &agent, const SDeepDriveRoadNetwork &roadNetwork, UBezierCurveComponent &bezierCmp, const SDeepDrivePathConfiguration &pathCfg);

	void setRoute(const SDeepDriveRoute &route);

	void advance(float deltaSeconds, float &speed, float &steering, float &brake);

	bool hasReachedDestination() const;

	float getRouteLength() const;

	float getDistanceAlongRoute() const;

	float getDistanceToCenterOfTrack() const;

	void showPath(UWorld *world);

private:

	ADeepDriveAgent 						&m_Agent;
	const SDeepDriveRoadNetwork				&m_RoadNetwork;
	
	UBezierCurveComponent					&m_BezierCurve;

	const SDeepDrivePathConfiguration		&m_PathConfiguration;
	DeepDriveBasePath						*m_curBasePath = 0;

};
