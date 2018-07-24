

#include "DeepDrivePluginPrivatePCH.h"
#include "Public/Simulation/Agent/Controllers/DeepDriveAgentRemoteAIController.h"
#include "Public/Simulation/DeepDriveSimulation.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"
#include "Public/Simulation/Misc/DeepDriveSplineTrack.h"
#include "Public/Simulation/Misc/DeepDriveRandomStream.h"

#include "Public/Server/Messages/DeepDriveServerSimulation.h"

#include "Components/SplineComponent.h"

ADeepDriveAgentRemoteAIController::ADeepDriveAgentRemoteAIController()
{
	m_ControllerName = "Remote AI Controller";
}

void ADeepDriveAgentRemoteAIController::OnConfigureSimulation(const deepdrive::server::SimulationConfiguration &configuration, bool initialConfiguration)
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
		m_Agent->SetControlValues(steering, throttle, brake, handbrake);
}

bool ADeepDriveAgentRemoteAIController::Activate(ADeepDriveAgent &agent)
{
	return initAgentOnTrack(agent) && ADeepDriveAgentControllerBase::Activate(agent);
}

bool ADeepDriveAgentRemoteAIController::ResetAgent()
{
	bool res = false;
	if(m_Agent)
	{
		resetAgentPosOnSpline(*m_Agent, m_Track->GetSpline(), m_StartDistance);
		res = true;
	}
	return res;
}


void ADeepDriveAgentRemoteAIController::Configure(const FDeepDriveRemoteAIControllerConfiguration &Configuration, int32 StartPositionSlot, ADeepDriveSimulation* DeepDriveSimulation)
{
	m_DeepDriveSimulation = DeepDriveSimulation;
	m_Track = Configuration.Track;
	m_StartDistance = StartPositionSlot >= 0 && StartPositionSlot < Configuration.StartDistances.Num() ? Configuration.StartDistances[StartPositionSlot] : -1.0f;
}
