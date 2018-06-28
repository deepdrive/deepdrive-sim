

#include "DeepDrivePluginPrivatePCH.h"
#include "Public/Simulation/Agent/Controllers/DeepDriveAgentRemoteAIController.h"
#include "Public/Simulation/DeepDriveSimulation.h"
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
