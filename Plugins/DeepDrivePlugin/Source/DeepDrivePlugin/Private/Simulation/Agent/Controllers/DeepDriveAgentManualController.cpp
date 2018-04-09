

#include "DeepDrivePluginPrivatePCH.h"
#include "Public/Simulation/Agent/Controllers/DeepDriveAgentManualController.h"

#include "Public/Simulation/Agent/DeepDriveAgent.h"


void ADeepDriveAgentManualController::Activate()
{
}

void ADeepDriveAgentManualController::Deactivate()
{
}

void ADeepDriveAgentManualController::MoveForward(float axisValue)
{
	if(m_Agent)
		m_Agent->SetThrottle(axisValue);
}

void ADeepDriveAgentManualController::MoveRight(float axisValue)
{
	if(m_Agent)
		m_Agent->SetSteering(axisValue);
}

void ADeepDriveAgentManualController::SetControlValues(float steering, float throttle, float brake, bool handbrake)
{

}
