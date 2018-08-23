

#include "DeepDrivePluginPrivatePCH.h"
#include "Public/Simulation/Agent/Controllers/DeepDriveAgentManualController.h"
#include "Public/Simulation/DeepDriveSimulation.h"
#include "Public/Simulation/Misc/DeepDriveRandomStream.h"
#include "Public/Simulation/Misc/DeepDriveSplineTrack.h"

#include "Public/Simulation/Agent/DeepDriveAgent.h"
#include "Components/SplineComponent.h"

ADeepDriveAgentManualController::ADeepDriveAgentManualController()
	:	ADeepDriveAgentControllerBase()
{
	m_ControllerName = "Manual Controller";
}

bool ADeepDriveAgentManualController::Activate(ADeepDriveAgent &agent, bool keepPosition)
{
	bool res = false;
	if(keepPosition || initAgentOnTrack(agent))
	{
		activateController(agent);
		res = true;
	}
	return res;
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

void ADeepDriveAgentManualController::Configure(const FDeepDriveManualControllerConfiguration &Configuration, int32 StartPositionSlot, ADeepDriveSimulation* DeepDriveSimulation)
{
	m_DeepDriveSimulation = DeepDriveSimulation;
	m_Track = Configuration.Track;
	m_StartDistance = StartPositionSlot >= 0 && StartPositionSlot < Configuration.StartDistances.Num() ? Configuration.StartDistances[StartPositionSlot] : -1.0f;
}

