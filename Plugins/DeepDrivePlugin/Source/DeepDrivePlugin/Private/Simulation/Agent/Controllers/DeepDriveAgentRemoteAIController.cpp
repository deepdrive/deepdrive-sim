

#include "DeepDrivePluginPrivatePCH.h"
#include "Public/Simulation/Agent/Controllers/DeepDriveAgentRemoteAIController.h"

#include "Public/Simulation/Agent/DeepDriveAgent.h"

ADeepDriveAgentRemoteAIController::ADeepDriveAgentRemoteAIController()
{
	m_ControllerName = "Remote AI Controller";
}

void ADeepDriveAgentRemoteAIController::SetControlValues(float steering, float throttle, float brake, bool handbrake)
{
	if (m_Agent)
		m_Agent->SetControlValues(steering, throttle, brake, handbrake);
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
