// 

#pragma once

#include "GameFramework/Actor.h"
#include "Components/SplineComponent.h"
#include "SplineTrajectory.generated.h"

UCLASS()
class DEEPDRIVE_API ASplineTrajectory : public AActor
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	ASplineTrajectory();

	// Called when the game starts or when spawned
	virtual void BeginPlay() override;
	
	// Called every frame
	virtual void Tick( float DeltaSeconds ) override;


	UPROPERTY(Category = Components, VisibleAnywhere, BlueprintReadWrite, meta = (AllowPrivateAccess = "true"))
	USplineComponent* Trajectory;
	
};
