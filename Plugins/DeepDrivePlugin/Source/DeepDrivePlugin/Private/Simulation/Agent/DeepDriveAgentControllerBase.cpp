

#include "DeepDrivePluginPrivatePCH.h"
#include "DeepDriveAgentControllerBase.h"


ADeepDriveAgentControllerBase::ADeepDriveAgentControllerBase()
{
}

void ADeepDriveAgentControllerBase::Possess(APawn *pawn)
{
	ADeepDriveAgent *agent = Cast<ADeepDriveAgent> (pawn);

	if(agent)
	{
		Super::Possess(pawn);
	}
}

void ADeepDriveAgentControllerBase::UnPossess()
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
