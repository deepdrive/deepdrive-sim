

#include "DeepDrivePluginPrivatePCH.h"
#include "DeepDriveAgentCityAIController.h"
#include "Public/Simulation/DeepDriveSimulation.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoadNetworkComponent.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoute.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"
#include "Private/Simulation/Agent/Controllers/DeepDriveAgentSpeedController.h"
#include "Private/Simulation/Agent/Controllers/DeepDriveAgentSteeringController.h"

ADeepDriveAgentCityAIController::ADeepDriveAgentCityAIController()
{
	m_ControllerName = "City AI Controller";

	//	for testing purposes
	m_isCollisionEnabled = true;
}

void ADeepDriveAgentCityAIController::Tick( float DeltaSeconds )
{
	if(m_Route && m_Agent && m_SpeedController && m_SteeringController)
	{
		m_Route->update(*m_Agent);

		if (m_hasActiveGuidance)
		{
			if (m_Route->getRemainingDistance() < 1000.0f)
				m_hasActiveGuidance = false;
			float desiredSpeed = m_DesiredSpeed;

			desiredSpeed = m_SpeedController->limitSpeedByTrack(desiredSpeed, 1.0f);
			m_SpeedController->update(DeltaSeconds, desiredSpeed, -1.0f, 0.0f);
			m_SteeringController->update(DeltaSeconds, desiredSpeed, 0.0f);
		}
		else
		{
			m_Agent->SetThrottle(0.0f);
			m_Agent->SetBrake(1.0f);
			m_Agent->SetSteering(0.0f);
		}
	}
}

bool ADeepDriveAgentCityAIController::Activate(ADeepDriveAgent &agent, bool keepPosition)
{
	bool res = false;
	// if(keepPosition || initAgentOnTrack(agent))
	{
		m_DeepDriveSimulation->RoadNetwork->PlaceAgentOnRoad(&agent, m_StartPos);

		m_Route = m_DeepDriveSimulation->RoadNetwork->CalculateRoute(FVector::OneVector, FVector::OneVector);
		//m_Route = m_DebugRoutes.Num() > 0 ? m_DeepDriveSimulation->RoadNetwork->CalculateRoute(m_DebugRoutes[0]) : 0;
		if (m_Route)
		{
			m_Route->convert(FVector::OneVector);

			m_SpeedController = new DeepDriveAgentSpeedController(m_Configuration.PIDThrottle, m_Configuration.PIDBrake);
			m_SpeedController->initialize(agent, *m_Route, m_Configuration.SafetyDistanceFactor);

			m_SteeringController = new DeepDriveAgentSteeringController(m_Configuration.PIDSteering);
			m_SteeringController->initialize(agent, *m_Route);
			m_hasActiveGuidance = true;
		}

		activateController(agent);
		res = true;
	}
	return res;
}

bool ADeepDriveAgentCityAIController::ResetAgent()
{
	bool res = false;
	if(m_Agent)
	{
        res = true;
		m_Agent->reset();
	}
	return res;
}

void ADeepDriveAgentCityAIController::Configure(const FDeepDriveCityAIControllerConfiguration &Configuration, int32 StartPositionSlot, ADeepDriveSimulation* DeepDriveSimulation)
{
	m_Configuration = Configuration;
	m_DeepDriveSimulation = DeepDriveSimulation;
	m_StartPos = Configuration.StartPositions[StartPositionSlot];

	for (auto &srcRoute : m_Configuration.Routes)
	{
		TArray<uint32> route;
		for(auto &linkProxy : srcRoute.Links)
		{
			uint32 linkId = m_DeepDriveSimulation->RoadNetwork->getRoadLink(linkProxy);
			if (linkId)
				route.Add(linkId);
		}
		m_DebugRoutes.Add(route);
	}

	m_DesiredSpeed = FMath::RandRange(Configuration.SpeedRange.X, Configuration.SpeedRange.Y);
}
