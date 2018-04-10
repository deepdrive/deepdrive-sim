

#include "DeepDrivePluginPrivatePCH.h"
#include "Public/Simulation/Agent/DeepDriveAgentControllerBase.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"

DEFINE_LOG_CATEGORY(LogDeepDriveAgentControllerBase);

ADeepDriveAgentControllerBase::ADeepDriveAgentControllerBase()
{
}

ADeepDriveAgentControllerBase::~ADeepDriveAgentControllerBase()
{
	UE_LOG(LogDeepDriveAgentSplineController, Log, TEXT("~ADeepDriveAgentControllerBase: %p sayz bye"), this );
}

bool ADeepDriveAgentControllerBase::Activate(ADeepDriveAgent &agent)
{
	m_Agent = &agent;
	Possess(m_Agent);
	return true;
}

void ADeepDriveAgentControllerBase::Deactivate()
{
}

void ADeepDriveAgentControllerBase::MoveForward(float axisValue)
{

}

void ADeepDriveAgentControllerBase::MoveRight(float axisValue)
{

}	

void ADeepDriveAgentControllerBase::SetControlValues(float steering, float throttle, float brake, bool handbrake)
{

}
