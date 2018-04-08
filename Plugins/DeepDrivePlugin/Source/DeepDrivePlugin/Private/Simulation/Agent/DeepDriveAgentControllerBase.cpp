

#include "DeepDrivePluginPrivatePCH.h"
#include "DeepDriveAgentControllerBase.h"


ADeepDriveAgentControllerBase::ADeepDriveAgentControllerBase()
{
}

void ADeepDriveAgentControllerBase::Possess(APawn *pawn)
{
	m_Agent = Cast<ADeepDriveAgent>(pawn);

	if (m_Agent)
	{
		Super::Possess(pawn);
		Activate();
	}
}

void ADeepDriveAgentControllerBase::UnPossess()
{
	if (m_Agent)
	{
		Deactivate();
		m_Agent = 0;
	}
}

void ADeepDriveAgentControllerBase::Activate()
{
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
