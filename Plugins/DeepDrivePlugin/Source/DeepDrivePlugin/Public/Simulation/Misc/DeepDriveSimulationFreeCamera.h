

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "DeepDriveSimulationFreeCamera.generated.h"

class UCameraComponent;


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
	float ForwardSpeed = 50.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Movement)
	float RightSpeed = 25.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Movement)
	float InterpolationSpeed = 2.0f;

	UFUNCTION(BlueprintCallable, Category = "Movement")
	void MoveForward(float AxisValue);

	UFUNCTION(BlueprintCallable, Category = "Movement")
	void MoveRight(float AxisValue);

	UFUNCTION(BlueprintCallable, Category = "Movement")
	void LookUp(float AxisValue);

	UFUNCTION(BlueprintCallable, Category = "Movement")
	void Turn(float AxisValue);

protected:

	UPROPERTY(EditDefaultsOnly, Category = Cameras)
	UCameraComponent					*FreeCamera = 0;

private:

	float					m_DesiredForward = 0.0f;
	float					m_curForward = 0.0f;

	float					m_DesiredRight = 0.0f;
	float					m_curRight = 0.0f;

	float					m_LookUp = 0.0f;
	float					m_Turn = 0.0f;

};

inline void ADeepDriveSimulationFreeCamera::MoveForward(float AxisValue)
{
	m_DesiredForward = AxisValue;
}

inline void ADeepDriveSimulationFreeCamera::MoveRight(float AxisValue)
{
	m_DesiredRight = AxisValue;
}

inline void ADeepDriveSimulationFreeCamera::LookUp(float AxisValue)
{
	m_LookUp = AxisValue;
}

inline void ADeepDriveSimulationFreeCamera::Turn(float AxisValue)
{
	m_Turn = AxisValue;
}

