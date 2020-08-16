

#pragma once

#include "CoreMinimal.h"
#include "Simulation/Agent/DeepDriveAgentControllerBase.h"
#include "DeepDriveAgentOneOffController.generated.h"

/**
 * 
 */
UCLASS()
class DEEPDRIVEPLUGIN_API ADeepDriveAgentOneOffController : public ADeepDriveAgentControllerBase
{
	GENERATED_BODY()
	
public:

	ADeepDriveAgentOneOffController();

	virtual bool Activate(ADeepDriveAgent &agent, bool keepPosition);

	virtual void MoveForward(float axisValue);

	virtual void MoveRight(float axisValue);

	virtual void Brake(float axisValue);

	void configure(ADeepDriveSimulation* deepDriveSim, const FTransform &transform, bool bindToRoad);

protected:

	FTransform			m_StartTransform;

	bool				m_bindToRoad;
};


inline void ADeepDriveAgentOneOffController::configure(ADeepDriveSimulation* deepDriveSim, const FTransform &transform, bool bindToRoad)
{
	m_DeepDriveSimulation = deepDriveSim;
	m_StartTransform = transform;
	m_bindToRoad = bindToRoad;
}

