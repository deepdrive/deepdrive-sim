

#pragma once

#include "GameFramework/Controller.h"
#include "DeepDriveAgentControllerBase.generated.h"


class ADeepDriveAgent;

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


	virtual void Activate();

	virtual void Deactivate();

	virtual void MoveForward(float axisValue);

	virtual void MoveRight(float axisValue);
	
	virtual void SetControlValues(float steering, float throttle, float brake, bool handbrake);
	
	
protected:

	ADeepDriveAgent							*m_Agent = 0;
};
