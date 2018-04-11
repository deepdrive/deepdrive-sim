

#include "DeepDrivePluginPrivatePCH.h"
#include "Public/Simulation/Agent/Controllers/DeepDriveAgentManualController.h"

#include "Public/Simulation/Agent/DeepDriveAgent.h"


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

