

#include "DeepDrivePluginPrivatePCH.h"
#include "Public/Simulation/Agent/Controllers/DeepDriveAgentRemoteAIController.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"
#include "Public/Simulation/Misc/DeepDriveSplineTrack.h"

#include "Components/SplineComponent.h"

ADeepDriveAgentRemoteAIController::ADeepDriveAgentRemoteAIController()
{
	m_ControllerName = "Remote AI Controller";
}

void ADeepDriveAgentRemoteAIController::SetControlValues(float steering, float throttle, float brake, bool handbrake)
{
	if (m_Agent)
		m_Agent->SetControlValues(steering, throttle, brake, handbrake);
}

bool ADeepDriveAgentRemoteAIController::Activate(ADeepDriveAgent &agent)
{
	if (m_Track)
	{
		m_Track->registerAgent(agent, m_Track->GetSpline()->FindInputKeyClosestToWorldLocation(agent.GetActorLocation()));
	}

	return ADeepDriveAgentControllerBase::Activate(agent);
}

bool ADeepDriveAgentRemoteAIController::ResetAgent()
{
	bool res = false;
	if(m_Agent)
	{
		m_Agent->reset();

		res = true;
	}
	return res;
}


void ADeepDriveAgentRemoteAIController::Configure(const FDeepDriveRemoteAIControllerConfiguration &Configuration, int32 StartPositionSlot)
{
	m_Track = Configuration.Track;
}
