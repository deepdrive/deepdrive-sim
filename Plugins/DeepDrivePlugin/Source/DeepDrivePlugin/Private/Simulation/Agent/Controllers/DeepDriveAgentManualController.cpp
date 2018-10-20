

#include "DeepDrivePluginPrivatePCH.h"
#include "Public/Simulation/Agent/Controllers/DeepDriveAgentManualController.h"
#include "Public/Simulation/DeepDriveSimulation.h"
#include "Public/Simulation/Misc/DeepDriveRandomStream.h"
#include "Public/Simulation/Misc/DeepDriveSplineTrack.h"

#include "Public/Simulation/Agent/DeepDriveAgent.h"
#include "Components/SplineComponent.h"

#include "WheeledVehicleMovementComponent.h"

ADeepDriveAgentManualController::ADeepDriveAgentManualController()
	:	ADeepDriveAgentControllerBase()
{
	m_ControllerName = "Manual Controller";

	//	for testing purposes
	m_isCollisionEnabled = true;
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
	{
		if(axisValue >= 0.0f)
		{
			m_Agent->GetVehicleMovementComponent()->SetTargetGear(1, false);
			m_Agent->SetBrake(0.0f);
		}
		else if( axisValue < 0.0f)
		{
			m_Agent->GetVehicleMovementComponent()->SetTargetGear(-1, false);
			m_Agent->SetBrake(m_Agent->GetVehicleMovementComponent()->GetForwardSpeed() > 0.0f);
		}
		else
		{
			m_Agent->GetVehicleMovementComponent()->SetTargetGear(0, false);
			m_Agent->SetBrake(1.0f);
		}

		m_Agent->SetThrottle(axisValue);
	}
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

