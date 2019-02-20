

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
			if (m_Route->getRemainingDistance() - m_Agent->getFrontBumperDistance() < 800.0f)
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
		UDeepDriveRoadNetworkComponent *roadNetwork = m_DeepDriveSimulation->RoadNetwork;

		if(m_StartIndex < 0 || m_StartIndex < m_Configuration.Routes.Num())
		{
			FVector start;
			if(m_StartIndex < 0)
			{

			}
			else
			{
				start = m_Configuration.Routes[m_StartIndex].Start;
				FVector dest = m_Configuration.Routes[m_StartIndex].Destination;
				m_Route = roadNetwork->CalculateRoute(start, dest);
			}

			if (m_Route)
			{
				m_Route->convert(start);
				m_Route->placeAgentAtStart(agent);

				m_SpeedController = new DeepDriveAgentSpeedController(m_Configuration.PIDThrottle, m_Configuration.PIDBrake);
				m_SpeedController->initialize(agent, *m_Route, m_Configuration.SafetyDistanceFactor);

				m_SteeringController = new DeepDriveAgentSteeringController(m_Configuration.PIDSteering);
				m_SteeringController->initialize(agent, *m_Route);
				m_hasActiveGuidance = true;
			}
			else
			{
				roadNetwork->PlaceAgentOnRoad(&agent, start);
			}

			res = true;
		}
		else if(m_StartIndex < m_Configuration.StartPositions.Num())
		{
			roadNetwork->PlaceAgentOnRoad(&agent, m_Configuration.StartPositions[m_StartIndex]);
			res = true;
		}

		if(res)
			activateController(agent);
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
	m_StartIndex = StartPositionSlot;

	m_DesiredSpeed = FMath::RandRange(Configuration.SpeedRange.X, Configuration.SpeedRange.Y);
}
