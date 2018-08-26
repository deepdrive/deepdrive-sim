

#include "DeepDrivePluginPrivatePCH.h"
#include "Public/Simulation/Agent/Controllers/DeepDriveAgentRemoteAIController.h"
#include "Public/Simulation/DeepDriveSimulation.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"
#include "Public/Simulation/Misc/DeepDriveSplineTrack.h"
#include "Public/Simulation/Misc/DeepDriveRandomStream.h"

#include "Public/Simulation/DeepDriveSimulationTypes.h"

#include "Components/SplineComponent.h"

ADeepDriveAgentRemoteAIController::ADeepDriveAgentRemoteAIController()
	:	ADeepDriveAgentControllerBase()
{
	m_ControllerName = "Remote AI Controller";
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
		m_Agent->SetControlValues(steering, throttle, brake, handbrake);
}

bool ADeepDriveAgentRemoteAIController::Activate(ADeepDriveAgent &agent, bool keepPosition)
{
	bool activated = false;
	if(keepPosition || initAgentOnTrack(agent))
	{
		activateController(agent);
		m_hasCollisionOccured = false;
		activated = true;
	}
	return true;
}

bool ADeepDriveAgentRemoteAIController::ResetAgent()
{
	bool res = false;
	if(m_Agent)
	{
		resetAgentPosOnSpline(*m_Agent, m_Track->GetSpline(), m_StartDistance);
		m_hasCollisionOccured = false;
		res = true;
	}
	return res;
}

void ADeepDriveAgentRemoteAIController::OnAgentCollision(AActor *OtherActor, const FHitResult &HitResult)
{
	FDateTime now(FDateTime::UtcNow());
	m_lastCollisionTimeStamp = FPlatformTime::Seconds();
	m_lastCollisionTimeUTC = FDateTime::UtcNow();
	m_hasCollisionOccured = true;
}



void ADeepDriveAgentRemoteAIController::Configure(const FDeepDriveRemoteAIControllerConfiguration &Configuration, int32 StartPositionSlot, ADeepDriveSimulation* DeepDriveSimulation)
{
	m_DeepDriveSimulation = DeepDriveSimulation;
	m_Track = Configuration.Track;
	m_StartDistance = StartPositionSlot >= 0 && StartPositionSlot < Configuration.StartDistances.Num() ? Configuration.StartDistances[StartPositionSlot] : -1.0f;
}
