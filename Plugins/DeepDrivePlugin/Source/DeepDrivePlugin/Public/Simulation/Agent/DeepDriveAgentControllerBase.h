

#pragma once

#include "GameFramework/Controller.h"
#include "DeepDriveAgentControllerBase.generated.h"

/**
 * 
 */
UCLASS()
class DEEPDRIVEPLUGIN_API ADeepDriveAgentControllerBase : public AController
{
	GENERATED_BODY()
	
public:

	ADeepDriveAgentControllerBase();

	virtual void Possess(APawn *pawn) override;

	virtual void UnPossess() override;



	virtual void MoveForward(float axisValue);

	virtual void MoveRight(float axisValue);
	
	virtual void SetControlValues(float steering, float throttle, float brake, bool handbrake);
	
	
};
