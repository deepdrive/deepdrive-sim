

#include "DeepDrivePluginPrivatePCH.h"
#include "Public/Simulation/Agent/Controllers/DeepDriveAgentManualController.h"
#include "Public/Simulation/DeepDriveSimulation.h"

#include "Public/Simulation/Agent/DeepDriveAgent.h"

ADeepDriveAgentManualController::ADeepDriveAgentManualController()
{
	m_ControllerName = "Manual Controller";
}

bool ADeepDriveAgentManualController::Activate(ADeepDriveAgent &agent)
{
	if (m_Track && m_DeepDriveSimulation)
	{
		m_Track->registerAgent(agent, m_Track->GetSpline()->FindInputKeyClosestToWorldLocation(agent.GetActorLocation()));

		if(m_StartDistance < 0.0f)
		{
			m_StartDistance = m_Track->getRandomDistanceAlongTrack(m_DeepDriveSimulation->getRandomStream());
			UE_LOG(LogDeepDriveAgentControllerBase, Log, TEXT("ADeepDriveAgentRemoteAIController::Activate random start distance %f"), m_StartDistance);
		}

		if(m_StartDistance >= 0.0f)
			resetAgentPosOnSpline(agent, m_Track->GetSpline(), m_StartDistance);
	}

	return m_StartDistance >= 0.0f && m_Track && ADeepDriveAgentControllerBase::Activate(agent);
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

