

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "DeepDriveSplineTrack.generated.h"

class USplineComponent;


/**
 * 
 */

UCLASS()
class DEEPDRIVEPLUGIN_API ADeepDriveSplineTrack	:	public AActor
{
	GENERATED_BODY()

public:

	ADeepDriveSplineTrack();

	~ADeepDriveSplineTrack();

	virtual void BeginPlay() override;


	void setBaseLocation(const FVector &baseLocation);

	FVector getLocationAhead(float distanceAhead, float sideOffset);

	float getSpeedLimit(float distanceAhead);


	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Track")
	TArray<FVector2D>		SpeedLimits;

	UFUNCTION(BlueprintCallable, Category = "Track")
	USplineComponent* GetSpline();

protected:

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Track")
	USplineComponent		*SplineTrack = 0;

private:

	float getInputKeyAhead(float distanceAhead);

	TArray<FVector2D>				m_SpeedLimits;

	FVector							m_BaseLocation;
	float							m_BaseKey = 0.0f;
};

inline USplineComponent* ADeepDriveSplineTrack::GetSpline()
{
	return SplineTrack;
}
