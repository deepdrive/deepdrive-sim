
#pragma once

#include "Private/Simulation/DeepDriveSimulationStateMachine.h"

#include "Public/Simulation/DeepDriveSimulationDefines.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveSimulationConfigureState, Log, All);

struct FDeepDriveScenarioConfiguration;

class DeepDriveSimulationConfigureState : public DeepDriveSimulationStateBase
{
public:

	DeepDriveSimulationConfigureState(DeepDriveSimulationStateMachine &stateMachine, UWorld *world);

	virtual ~DeepDriveSimulationConfigureState();

	virtual void enter(ADeepDriveSimulation &deepDriveSim);

	virtual void update(ADeepDriveSimulation &deepDriveSim, float dT);

	virtual void exit(ADeepDriveSimulation &deepDriveSim);

	void setConfiguration(const FDeepDriveScenarioConfiguration &configuration);

private:

	UWorld									*m_World = 0;
	FDeepDriveScenarioConfiguration			m_Configuration;
};


inline void DeepDriveSimulationConfigureState::setConfiguration(const FDeepDriveScenarioConfiguration &configuration)
{
	m_Configuration = configuration;
}
