

#include "DeepDrivePluginPrivatePCH.h"
#include "Public/Simulation/Agent/Controllers/DeepDriveAgentRemoteAIController.h"
#include "Public/Simulation/DeepDriveSimulation.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoadNetworkComponent.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoute.h"
#include "Public/Simulation/Misc/DeepDriveSplineTrack.h"
#include "Public/Simulation/Misc/DeepDriveRandomStream.h"

#include "Public/Simulation/DeepDriveSimulationTypes.h"

#include "Components/SplineComponent.h"

ADeepDriveAgentRemoteAIController::ADeepDriveAgentRemoteAIController()
	:	ADeepDriveAgentControllerBase()
{
	m_ControllerName = "Remote AI Controller";
	m_isCollisionEnabled = true;
}

void ADeepDriveAgentRemoteAIController::OnConfigureSimulation(const SimulationConfiguration &configuration, bool initialConfiguration)
{
	UE_LOG(LogDeepDriveAgentControllerBase, Log, TEXT("ADeepDriveAgentRemoteAIController Reconfigure %f"), configuration.agent_start_location);
	if (m_Agent)
	{
		m_StartDistance = configuration.agent_start_location;
		initAgentOnTrack(*m_Agent);
	}
}


void ADeepDriveAgentRemoteAIController::SetControlValues(float steering, float throttle, float brake, bool handbrake)
{
	if (m_Agent)
	{
		m_Agent->SetControlValues(steering, throttle, brake, handbrake);
		m_Agent->setIsGameDriving(false);
	}
}


bool ADeepDriveAgentRemoteAIController::Activate(ADeepDriveAgent &agent, bool keepPosition)
{
	bool activated = false;
	switch(m_OperationMode)
	{
		case OperationMode::Standard:
			if(keepPosition || initAgentOnTrack(agent))
			{
				activateController(agent);
				m_hasCollisionOccured = false;
			}
			break;

		case OperationMode::Scenario:
			{
				UDeepDriveRoadNetworkComponent *roadNetwork = m_DeepDriveSimulation->RoadNetwork;
				FVector start = m_ScenarionConfiguration.StartPosition;
				m_Route = roadNetwork->CalculateRoute(start, m_ScenarionConfiguration.EndPosition);
				if (m_Route)
				{
					m_Route->convert(start);
					m_Route->placeAgentAtStart(agent);
				}
				else
				{
					roadNetwork->PlaceAgentOnRoad(&agent, start, true);
				}
				activated = true;
			}
			break;
	}
	return true;
}

bool ADeepDriveAgentRemoteAIController::ResetAgent()
{
	bool res = false;
	if(m_Agent)
	{
		UE_LOG(LogDeepDriveAgentControllerBase, Log, TEXT("Reset Agent at %d"), m_StartPositionSlot);
		if(m_StartPositionSlot < 0)
			m_StartDistance = m_Track->getRandomDistanceAlongTrack(*m_DeepDriveSimulation->GetRandomStream(FName("AgentPlacement")));

		resetAgentPosOnSpline(*m_Agent, m_Track->GetSpline(), m_StartDistance);
		m_Agent->reset();
		m_hasCollisionOccured = false;
		res = true;
	}
	return res;
}


void ADeepDriveAgentRemoteAIController::Configure(const FDeepDriveRemoteAIControllerConfiguration &Configuration, int32 StartPositionSlot, ADeepDriveSimulation* DeepDriveSimulation)
{
	m_DeepDriveSimulation = DeepDriveSimulation;
	m_OperationMode = OperationMode::Standard;

	m_Track = Configuration.Track;
	m_StartPositionSlot = StartPositionSlot;
	m_StartDistance = StartPositionSlot >= 0 && StartPositionSlot < Configuration.StartDistances.Num() ? Configuration.StartDistances[StartPositionSlot] : -1.0f;
}

void ADeepDriveAgentRemoteAIController::ConfigureScenario(const FDeepDriveRemoteAIControllerConfiguration &Configuration, const FDeepDriveAgentScenarioConfiguration &ScenarioConfiguration, ADeepDriveSimulation* DeepDriveSimulation)
{
	m_DeepDriveSimulation = DeepDriveSimulation;

	m_OperationMode = OperationMode::Scenario;
	m_ScenarionConfiguration.StartPosition = ScenarioConfiguration.StartPosition;
	m_ScenarionConfiguration.EndPosition = ScenarioConfiguration.EndPosition;
}
