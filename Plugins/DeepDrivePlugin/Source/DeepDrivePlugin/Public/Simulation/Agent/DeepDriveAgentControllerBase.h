

#pragma once

#include "GameFramework/Controller.h"
#include "DeepDriveAgentControllerBase.generated.h"


DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveAgentControllerBase, Log, All);

class ADeepDriveAgent;

/**
 * 
 */
UCLASS(Blueprintable, BlueprintType)
class DEEPDRIVEPLUGIN_API ADeepDriveAgentControllerBase : public AController
{
	GENERATED_BODY()
	
public:

	~ADeepDriveAgentControllerBase();

	virtual bool Activate(ADeepDriveAgent &agent);

	virtual void Deactivate();

	virtual void MoveForward(float axisValue);

	virtual void MoveRight(float axisValue);
	
	virtual void SetControlValues(float steering, float throttle, float brake, bool handbrake);
	
	virtual bool ResetAgent();
	
	const FString& getControllerName() const;

protected:

	ADeepDriveAgent						*m_Agent = 0;

	FString								m_ControllerName = "DeepDriveAgentControllerBase";
	bool								m_isGameDriving = false;
};


inline const FString& ADeepDriveAgentControllerBase::getControllerName() const
{
	return m_ControllerName;
}