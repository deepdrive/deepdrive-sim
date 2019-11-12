

#include "DeepDrivePluginPrivatePCH.h"
#include "DeepDriveAgentRemoteTrafficAIController.h"
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

ADeepDriveAgentRemoteTrafficAIController::ADeepDriveAgentRemoteTrafficAIController()
{
	m_ControllerName = "Remote Traffic AI Controller";
	//	for testing purposes
	m_isCollisionEnabled = true;

	m_BezierCurve = CreateDefaultSubobject<UBezierCurveComponent>(TEXT("BezierCurve"));
}

void ADeepDriveAgentRemoteTrafficAIController::SetControlValues(float steering, float throttle, float brake, bool handbrake)
{
	if (m_Agent)
	{
		m_Agent->SetControlValues(steering, throttle, brake, handbrake);
		m_Agent->setIsGameDriving(false);
	}
}



bool ADeepDriveAgentRemoteTrafficAIController::updateAgentOnPath( float DeltaSeconds )
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

	return m_PathPlanner->isCloseToEnd(800.0f);
}

bool ADeepDriveAgentRemoteTrafficAIController::Activate(ADeepDriveAgent &agent, bool keepPosition)
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
					
					UE_LOG(LogDeepDriveAgentControllerBase, Log, TEXT("ADeepDriveAgentRemoteTrafficAIController::Activate Random start pos %s"), *(route.Start.ToString()) );

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

	if (route.Links.Num() > 0)
	{
		//route->convert(start);
		//route->placeAgentAtStart(agent);

		m_SpeedController = new DeepDriveAgentSpeedController(m_Configuration.PIDThrottle, m_Configuration.PIDBrake);
		m_SpeedController->initialize(agent);

		m_PathConfiguration = new SDeepDrivePathConfiguration;
		m_PathConfiguration->PIDSteering = m_Configuration.PIDSteering;
		m_PathPlanner = new DeepDrivePathPlanner(agent, roadNetwork->getRoadNetwork(), *m_BezierCurve, *m_PathConfiguration);

		DeepDriveManeuverCalculator *manCalc = m_DeepDriveSimulation->getManeuverCalculator();
		if(manCalc)
			manCalc->calculate(route, agent);
		m_PathPlanner->setRoute(route);
		
		{
			float speed = 0.0f;
			float brake = 1.0f;
			float heading = 0.0f;
			m_PathPlanner->advance(0.0f, speed, heading, brake);
		}

		res = true;
		m_State = ActiveRouteGuidance;
		m_Route = 0;
	}
	else
	{
		roadNetwork->PlaceAgentOnRoad(&agent, route.Start, true);
	}

	if(res)
		activateController(agent);

	return res;
}

bool ADeepDriveAgentRemoteTrafficAIController::ResetAgent()
{
	bool res = false;
	if(m_Agent)
	{
        res = true;
		m_Agent->reset();
	}
	return res;
}

float ADeepDriveAgentRemoteTrafficAIController::getDistanceAlongRoute()
{
	return m_PathPlanner && m_Agent ? m_PathPlanner->getDistanceAlongRoute() : 0.0f;
}

float ADeepDriveAgentRemoteTrafficAIController::getRouteLength()
{
	return m_PathPlanner && m_Agent ? m_PathPlanner->getRouteLength() : 0.0f;
}

float ADeepDriveAgentRemoteTrafficAIController::getDistanceToCenterOfTrack()
{
	return m_PathPlanner && m_Agent ? m_PathPlanner->getDistanceToCenterOfTrack() : 0.0f;
}

void ADeepDriveAgentRemoteTrafficAIController::Configure(const FDeepDriveTrafficAIControllerConfiguration &Configuration, int32 StartPositionSlot, ADeepDriveSimulation* DeepDriveSimulation)
{
	m_Configuration = Configuration;
	m_DeepDriveSimulation = DeepDriveSimulation;
	m_StartIndex = StartPositionSlot;

	m_OperationMode = OperationMode::Standard;

	m_DesiredSpeed = FMath::RandRange(Configuration.SpeedRange.X, Configuration.SpeedRange.Y);
}

void ADeepDriveAgentRemoteTrafficAIController::ConfigureScenario(const FDeepDriveTrafficAIControllerConfiguration &Configuration, const FDeepDriveAgentScenarioConfiguration &ScenarioConfiguration, ADeepDriveSimulation* DeepDriveSimulation)
{
	m_Configuration = Configuration;
	m_DeepDriveSimulation = DeepDriveSimulation;
	m_StartIndex = -1;

	m_OperationMode = OperationMode::Scenario;
	m_ScenarionConfiguration.StartPosition = ScenarioConfiguration.StartPosition;
	m_ScenarionConfiguration.EndPosition = ScenarioConfiguration.EndPosition;

	m_DesiredSpeed = ScenarioConfiguration.MaxSpeed;
}

float ADeepDriveAgentRemoteTrafficAIController::checkForObstacle(float maxDistance)
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

float ADeepDriveAgentRemoteTrafficAIController::calculateSafetyDistance()
{
	const float brakingDeceleration = 400.0f;
	const float curSpeed = m_Agent->getSpeed();
	const float safetyDistance = curSpeed * curSpeed / (2.0f * brakingDeceleration);

	return FMath::Max(100.0f, safetyDistance);
}
