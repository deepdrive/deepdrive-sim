

#pragma once

#include "Simulation/Agent/DeepDriveAgentControllerBase.h"
#include "DeepDriveAgentManualController.generated.h"

/**
 * 
 */
UCLASS()
class DEEPDRIVEPLUGIN_API ADeepDriveAgentManualController : public ADeepDriveAgentControllerBase
{
	GENERATED_BODY()
	
public:


	virtual void MoveForward(float axisValue);

	virtual void MoveRight(float axisValue);

	virtual void SetControlValues(float steering, float throttle, float brake, bool handbrake);

	
	
};
