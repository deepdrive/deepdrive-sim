

#pragma once

#include "Simulation/Agent/DeepDriveAgentControllerBase.h"
#include "DeepDriveAgentRemoteAIController.generated.h"

/**
 * 
 */
UCLASS()
class DEEPDRIVEPLUGIN_API ADeepDriveAgentRemoteAIController : public ADeepDriveAgentControllerBase
{
	GENERATED_BODY()
	
public:


	virtual void SetControlValues(float steering, float throttle, float brake, bool handbrake);
	
};
