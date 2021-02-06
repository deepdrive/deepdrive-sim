
#pragma once

#include "Simulation/DeepDriveSimulationStateMachine.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveSimulationResetState, Log, All);

class DeepDriveSimulationResetState : public DeepDriveSimulationStateBase
{
public:

	DeepDriveSimulationResetState(DeepDriveSimulationStateMachine &stateMachine, UWorld *world);

	virtual ~DeepDriveSimulationResetState();

	virtual void enter(ADeepDriveSimulation &deepDriveSim);

	virtual void update(ADeepDriveSimulation &deepDriveSim, float dT);

	virtual void exit(ADeepDriveSimulation &deepDriveSim);

	void setActivateAdditionalAgents(bool activateAgents);

private:

	UWorld		*m_World = 0;
	bool		m_ActivateAdditionalAgents = true;

};


inline void DeepDriveSimulationResetState::setActivateAdditionalAgents(bool activateAgents)
{
	m_ActivateAdditionalAgents = activateAgents;
}
