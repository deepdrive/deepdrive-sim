

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "DeepDriveSimulationFreeCamera.generated.h"

UCLASS()
class DEEPDRIVEPLUGIN_API ADeepDriveSimulationFreeCamera : public AActor
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	ADeepDriveSimulationFreeCamera();

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Movement)
	float ForwardSpeed = 200.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Movement)
	float RightSpeed = 100.0f;

	void MoveForward(float AxisValue);
	void MoveRight(float AxisValue);
	void LookUp(float AxisValue);
	void Turn(float AxisValue);

protected:

	UPROPERTY(EditDefaultsOnly, Category = Cameras)
	UCameraComponent					*FreeCamera = 0;

	
};
