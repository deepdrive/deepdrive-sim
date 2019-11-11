#pragma once

#include "CoreMinimal.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoadNetwork.h"
#include "Private/Simulation/Traffic/Path/DeepDrivePathDefines.h"

struct SDeepDriveRoadNetwork;

class DeepDrivePartialPath;
class ADeepDriveAgent;

class UBezierCurveComponent;
struct SDeepDrivePathConfiguration;
struct SDeepDriveManeuver;

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveBasePath, Log, All);

class DeepDriveBasePath
{
public:

    DeepDriveBasePath(ADeepDriveAgent &agent, const SDeepDriveRoadNetwork &roadNetwork,  UBezierCurveComponent &bezierCmp, const SDeepDrivePathConfiguration &pathCfg);

	~DeepDriveBasePath();

	void setRoute(const SDeepDriveRoute &route);

	void advance(float deltaSeconds, float &speed, float &steering, float &brake);

	bool isCloseToEnd(float distanceFromEnd) const;

	float getRouteLength() const;

	float getDistanceAlongRoute() const;

	float getDistanceToCenterOfTrack() const;

	void showPath(UWorld *world);

private:

	void convertRouteLinks();

	void extractCrossTrafficRoads(SDeepDriveManeuver &maneuver, const SDeepDriveJunction &junction);

	DeepDrivePartialPath* createPartialPath();
	float addRoadSegment(const SDeepDriveRoadSegment &segment,float carryOverDistance);

	ADeepDriveAgent 						&m_Agent;
	const SDeepDriveRoadNetwork				&m_RoadNetwork;
	UBezierCurveComponent					&m_BezierCurve;

	SDeepDriveRoute							m_Route;

	TArray<SDeepDriveBasePathSegment>		m_RouteSegments;

	const SDeepDrivePathConfiguration		&m_PathConfiguration;
	DeepDrivePartialPath					*m_PartialPath = 0;

};
