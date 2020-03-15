

#include "DeepDrivePluginPrivatePCH.h"
#include "DeepDriveAgentTrafficAIController.h"
#include "Public/Simulation/DeepDriveSimulation.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoadNetworkComponent.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoute.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"
#include "Private/Simulation/Agent/Controllers/DeepDriveAgentSpeedController.h"
#include "Private/Simulation/Agent/Controllers/DeepDriveAgentSteeringController.h"
#include "Public/Simulation/Misc/BezierCurveComponent.h"

#include "Private/Simulation/Traffic/Path/DeepDrivePathDefines.h"
#include "Private/Simulation/Traffic/Path/DeepDrivePathPlanner.h"
#include "Simulation/Traffic/Maneuver/DeepDriveManeuverCalculator.h"

#include "WheeledVehicleMovementComponent.h"

ADeepDriveAgentTrafficAIController::ADeepDriveAgentTrafficAIController()
{
	m_ControllerName = "Traffic AI Controller";

	//	for testing purposes
	m_isCollisionEnabled = true;

	m_BezierCurve = CreateDefaultSubobject<UBezierCurveComponent>(TEXT("BezierCurve"));
}

bool ADeepDriveAgentTrafficAIController::updateAgentOnPath( float DeltaSeconds )
{
	float speed = m_DesiredSpeed;
	float brake = 1.0f;
	float heading = 0.0f;
	m_PathPlanner->advance(DeltaSeconds, speed, heading, brake);

	const float safetyDistance = calculateSafetyDistance();
	float dist2Obstacle = checkForObstacle(safetyDistance);
	if(dist2Obstacle >= 0.0f)
		speed *= FMath::SmoothStep(300.0f, FMath::Max(500.0f, safetyDistance * 0.8f), dist2Obstacle);

	brake = speed > 0.0f ? 0.0f : 1.0f;

	m_SpeedController->update(DeltaSeconds, speed, brake);

	m_Agent->SetSteering(heading);

	// m_SteeringController->update(DeltaSeconds, heading);

	const bool hasReachedDestination = m_PathPlanner->hasReachedDestination();

	return hasReachedDestination;
}

void ADeepDriveAgentTrafficAIController::Tick( float DeltaSeconds )
{
	if(m_InputTimer > 0.0f)
	{
		m_InputTimer -= DeltaSeconds;
	}
	else if(m_Agent && m_SpeedController && !m_isRemotelyControlled)
	{
		m_Agent->GetVehicleMovementComponent()->SetTargetGear(1, false);
		switch (m_State)
		{
			case ActiveRouteGuidance:
				{
					const bool hasReachedDestination  = updateAgentOnPath(DeltaSeconds);

					if (hasReachedDestination)
					{
						m_State = m_StartIndex < 0 && m_OperationMode == OperationMode::Standard ? Waiting : Idle;
						m_Agent->SetThrottle(0.0f);
						m_Agent->SetBrake(1.0f);
						m_Agent->SetSteering(0.0f);

						m_WaitTimer = FMath::RandRange(3.0f, 4.0f);
					}

					if(m_showPath)
						m_PathPlanner->showPath(GetWorld());
				}
				break;

			case Waiting:
				m_WaitTimer -= DeltaSeconds;
				if(m_WaitTimer <= 0.0f)
				{
					FVector start = m_Agent->GetActorLocation();
					FVector destination;
					UE_LOG(LogDeepDriveAgentControllerBase, Log, TEXT("ADeepDriveAgentTrafficAIController::Activate Random start pos %s"), *(start.ToString()) );

					SDeepDriveRoute route;

					route.Links = m_DeepDriveSimulation->RoadNetwork->calculateRandomRoute(start, destination);
					if (route.Links.Num() > 0)
					{
						route.Start = start;
						route.Destination = destination;
						DeepDriveManeuverCalculator *manCalc = m_DeepDriveSimulation->getManeuverCalculator();
						if(manCalc)
							manCalc->calculate(route, *m_Agent);
						m_PathPlanner->setRoute(route);

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

bool ADeepDriveAgentTrafficAIController::Activate(ADeepDriveAgent &agent, bool keepPosition)
{
	bool res = false;

	UDeepDriveRoadNetworkComponent *roadNetwork = m_DeepDriveSimulation->RoadNetwork;

	m_showPath = roadNetwork->ShowRoutes;

	SDeepDriveRoute route;

	switch(m_OperationMode)
	{
		case OperationMode::Standard:
			if(m_StartIndex < 0 || m_StartIndex < m_Configuration.Routes.Num())
			{
				if(m_StartIndex < 0)
				{
					uint32_t fromLinkId = 0;
					do
					{
						fromLinkId = roadNetwork->getRandomRoadLink(false, true);
						if(fromLinkId)
							route.Start = roadNetwork->getRoadNetwork().getLocationOnLink(fromLinkId, EDeepDriveLaneType::MAJOR_LANE, 0.25f);
	
					} while(fromLinkId == 0 ||	m_DeepDriveSimulation->isLocationOccupied(route.Start, 300.0f));
					
					UE_LOG(LogDeepDriveAgentControllerBase, Log, TEXT("ADeepDriveAgentTrafficAIController::Activate Random start pos %s"), *(route.Start.ToString()) );

					roadNetwork->PlaceAgentOnRoad(&agent, route.Start, true);
					route.Links = roadNetwork->calculateRandomRoute(route.Start, route.Destination);
				}
				else
				{
					route.Start = m_Configuration.Routes[m_StartIndex].Start;
					route.Destination = m_Configuration.Routes[m_StartIndex].Destination;
					roadNetwork->PlaceAgentOnRoad(&agent, route.Start, true);
					route.Links = roadNetwork->CalculateRoute(route.Start, route.Destination);
				}
			}
			else if(m_StartIndex < m_Configuration.StartPositions.Num())
			{
				roadNetwork->PlaceAgentOnRoad(&agent, m_Configuration.StartPositions[m_StartIndex], true);
			}
			break;

		case OperationMode::Scenario:
			route.Start = m_ScenarionConfiguration.StartPosition;
			route.Destination = m_ScenarionConfiguration.EndPosition;
			roadNetwork->PlaceAgentOnRoad(&agent, route.Start, true);
			route.Links = roadNetwork->CalculateRoute(route.Start, route.Destination);
			break;
	}

	bool firstActivate = false;
	if (route.Links.Num() > 0)
	{
		if(m_SpeedController == 0)
		{
			m_SpeedController = new DeepDriveAgentSpeedController(m_Configuration.PIDThrottle, m_Configuration.PIDBrake);
			m_SpeedController->initialize(agent);
			firstActivate = true;
		}

		if(m_PathConfiguration == 0)
		{
			m_PathConfiguration = new SDeepDrivePathConfiguration;
			m_PathConfiguration->PIDSteering = m_Configuration.PIDSteering;
		}
		if(m_PathPlanner == 0)
			m_PathPlanner = new DeepDrivePathPlanner(agent, roadNetwork->getRoadNetwork(), *m_BezierCurve, *m_PathConfiguration);

		DeepDriveManeuverCalculator *manCalc = m_DeepDriveSimulation->getManeuverCalculator();
		if(manCalc)
			manCalc->calculate(route, agent);
		m_PathPlanner->setRoute(route);

		res = true;
		if (firstActivate)
			activateController(agent);
		m_State = ActiveRouteGuidance;
		m_Route = 0;
	}
	else
	{
		if (roadNetwork->getRoadNetwork().Junctions.Num() > 0)
			roadNetwork->PlaceAgentOnRoad(&agent, route.Start, true);

		res = true;
		activateController(agent);
	}

	return res;
}

void ADeepDriveAgentTrafficAIController::Reset()
{
	Activate(*m_Agent, false);
}

bool ADeepDriveAgentTrafficAIController::ResetAgent()
{
	bool res = false;
	if(m_Agent)
	{
        res = true;
		m_Agent->reset();
	}
	return res;
}

float ADeepDriveAgentTrafficAIController::getDistanceAlongRoute()
{
	return m_PathPlanner && m_Agent ? m_PathPlanner->getDistanceAlongRoute() : 0.0f;
}

float ADeepDriveAgentTrafficAIController::getRouteLength()
{
	return m_PathPlanner && m_Agent ? m_PathPlanner->getRouteLength() : 0.0f;
}

float ADeepDriveAgentTrafficAIController::getDistanceToCenterOfTrack()
{
	return m_PathPlanner && m_Agent ? m_PathPlanner->getDistanceToCenterOfTrack() : 0.0f;
}

void ADeepDriveAgentTrafficAIController::Configure(const FDeepDriveTrafficAIControllerConfiguration &Configuration, int32 StartPositionSlot, ADeepDriveSimulation* DeepDriveSimulation)
{
	m_Configuration = Configuration;
	m_DeepDriveSimulation = DeepDriveSimulation;
	m_StartIndex = StartPositionSlot;

	m_OperationMode = OperationMode::Standard;

	m_DesiredSpeed = FMath::RandRange(Configuration.SpeedRange.X, Configuration.SpeedRange.Y);
}

void ADeepDriveAgentTrafficAIController::ConfigureScenario(const FDeepDriveTrafficAIControllerConfiguration &Configuration, const FDeepDriveAgentScenarioConfiguration &ScenarioConfiguration, ADeepDriveSimulation* DeepDriveSimulation)
{
	m_Configuration = Configuration;
	m_DeepDriveSimulation = DeepDriveSimulation;
	m_StartIndex = -1;

	m_OperationMode = OperationMode::Scenario;
	m_ScenarionConfiguration.StartPosition = ScenarioConfiguration.StartPosition;
	m_ScenarionConfiguration.EndPosition = ScenarioConfiguration.EndPosition;

	m_DesiredSpeed = ScenarioConfiguration.MaxSpeed;
}

float ADeepDriveAgentTrafficAIController::checkForObstacle(float maxDistance)
{
	float distance = -1.0f;

	FHitResult hitResult;

	FVector forward = m_Agent-> GetActorForwardVector();
	FVector start = m_Agent->GetActorLocation() + forward * m_Agent->getFrontBumperDistance() + FVector(0.0f, 0.0f, 100.0f);
	FVector end = start + forward * FMath::Max(maxDistance, m_Agent->getFrontBumperDistance());

	// DrawDebugLine(GetWorld(), start, end, FColor::Green, false, 0.0f, 100, 4.0f);

	FCollisionQueryParams params;
	params.AddIgnoredActor(m_Agent);
	if(GetWorld()->LineTraceSingleByChannel(hitResult, start, end, ECollisionChannel::ECC_Visibility, params, FCollisionResponseParams() ))
	{
		distance = hitResult.Distance;
		// DrawDebugLine(GetWorld(), start, start + forward * distance, FColor::Red, false, 0.0f, 100, 6.0f);
	}

	return distance;
}

float ADeepDriveAgentTrafficAIController::calculateSafetyDistance()
{
	const float brakingDeceleration = 400.0f;
	const float curSpeed = m_Agent->getSpeed();
	const float safetyDistance = curSpeed * curSpeed / (2.0f * brakingDeceleration);

	return FMath::Max(100.0f, safetyDistance);
}
