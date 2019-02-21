

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

		switch(m_State)
		{
			case ActiveRouteGuidance:
				{
					if (m_Route->getRemainingDistance() - m_Agent->getFrontBumperDistance() < 800.0f)
					{
						m_State = m_StartIndex < 0 ? Waiting : Idle;
						m_WaitTimer = FMath::RandRange(3.0f, 4.0f);

					}
					float desiredSpeed = m_DesiredSpeed;

					desiredSpeed = m_SpeedController->limitSpeedByTrack(desiredSpeed, 1.0f);
					m_SpeedController->update(DeltaSeconds, desiredSpeed, -1.0f, 0.0f);
					m_SteeringController->update(DeltaSeconds, desiredSpeed, 0.0f);
				}
				break;

			case Waiting:
				m_WaitTimer -= DeltaSeconds;
				if(m_WaitTimer <= 0.0f)
				{
					FVector start = m_Agent->GetActorLocation();
					UE_LOG(LogDeepDriveAgentControllerBase, Log, TEXT("ADeepDriveAgentCityAIController::Activate Random start pos %s"), *(start.ToString()) );

					ADeepDriveRoute *route = m_DeepDriveSimulation->RoadNetwork->calculateRandomRoute(start);
					if (route)
					{
						route->convert(start);

						m_SteeringController->setRoute(*route);
						m_SpeedController->setRoute(*route);
						route->update(*m_Agent);

						m_Route->Destroy();
						m_Route = route;

						m_State = ActiveRouteGuidance;
					}
					else
					{
						m_State = Idle;
					}
				}			//	fall through

			case Idle:
				m_Agent->SetThrottle(0.0f);
				m_Agent->SetBrake(1.0f);
				m_Agent->SetSteering(0.0f);
				break;
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
				start = roadNetwork->GetRandomLocation(EDeepDriveLaneType::MAJOR_LANE, -1);
				UE_LOG(LogDeepDriveAgentControllerBase, Log, TEXT("ADeepDriveAgentCityAIController::Activate Random start pos %s"), *(start.ToString()) );

				m_Route = roadNetwork->calculateRandomRoute(start);
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
				m_State = ActiveRouteGuidance;
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
