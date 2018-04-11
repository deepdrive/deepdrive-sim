

#pragma once

#include "Simulation/Agent/DeepDriveAgentControllerBase.h"
#include "Components/SplineComponent.h"
#include "DeepDriveAgentSplineController.generated.h"


DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveAgentSplineController, Log, All);

/**
 * 
 */
UCLASS()
class DEEPDRIVEPLUGIN_API ADeepDriveAgentSplineController : public ADeepDriveAgentControllerBase
{
	GENERATED_BODY()
	
public:

	virtual void Tick( float DeltaSeconds ) override;

	virtual bool Activate(ADeepDriveAgent &agent);

private:

	USplineComponent		*m_Spline = 0;


	float getClosestDistanceOnSpline(const FVector &location);



	/**
		old implementation taken for ACar
	*/

	void MoveAlongSpline();

	bool UpdateSplineProgress();

	bool searchAlongSpline(FVector CurrentLocation, int step, float distToCurrent, float& distanceAlongRoute);

	bool getDistanceAlongSplineAtLocationWithStep(FVector CurrentLocation, unsigned int step, float& distanceAlongRoute);

	void GetDistanceAlongRouteAtLocation(FVector CurrentLocation);


	float					m_DistanceAlongRoute = 0.0f;
	float					m_DistanceToCenterOfLane = 0.0f;

	float					m_WaypointDistanceAlongSpline = 0.0f;
	float					m_WaypointStep = 400.0f;
	float					m_CloseDistanceThreshold = 1500.0f;

};
