
#include "Private/Simulation/Traffic/Path/DeepDrivePathPlanner.h"
#include "Private/Simulation/Traffic/Path/DeepDriveBasePath.h"


DeepDrivePathPlanner::DeepDrivePathPlanner(ADeepDriveAgent &agent, const SDeepDriveRoadNetwork &roadNetwork, UBezierCurveComponent &bezierCmp, const SDeepDrivePathConfiguration &pathCfg)
    :   m_Agent(agent)
    ,   m_RoadNetwork(roadNetwork)
	,	m_BezierCurve(bezierCmp)
    ,   m_PathConfiguration(pathCfg)
{
}

void DeepDrivePathPlanner::setRoute(const SDeepDriveRoute &route)
{
    delete m_curBasePath;

    m_curBasePath = new DeepDriveBasePath(m_Agent, m_RoadNetwork, m_BezierCurve, m_PathConfiguration);
    
    if(m_curBasePath)
        m_curBasePath->setRoute(route);
}

void DeepDrivePathPlanner::advance(float deltaSeconds, float &speed, float &steering, float &brake)
{
    if(m_curBasePath)
    {
        m_curBasePath->advance(deltaSeconds, speed, steering, brake);
    }
}

bool DeepDrivePathPlanner::hasReachedDestination() const
{
    return m_curBasePath ? m_curBasePath->hasReachedDestination() : true;
}

float DeepDrivePathPlanner::getRouteLength() const
{
    return m_curBasePath ? m_curBasePath->getRouteLength() : 0.0f;
}

float DeepDrivePathPlanner::getDistanceAlongRoute() const
{
    return m_curBasePath ? m_curBasePath->getDistanceAlongRoute() : 0.0f;
}

float DeepDrivePathPlanner::getDistanceToCenterOfTrack() const
{
    return m_curBasePath ? m_curBasePath->getDistanceToCenterOfTrack() : 0.0f;
}

void DeepDrivePathPlanner::showPath(UWorld *world)
{
    if(m_curBasePath)
        m_curBasePath->showPath(world);
}
