

#pragma once

#include "GameFramework/Actor.h"

#include "Public/Server/IDeepDriveServerProxy.h"
#include "Public/Simulation/DeepDriveSimulationDefines.h"

#include "DeepDriveSimulation.generated.h"

UCLASS()
class DEEPDRIVEPLUGIN_API ADeepDriveSimulation	:	public AActor
												,	public IDeepDriveServerProxy
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	ADeepDriveSimulation();

	// Called when the game starts or when spawned
	virtual void BeginPlay() override;
	
	// Called every frame
	virtual void Tick( float DeltaSeconds ) override;

	
	/**
	*		IDeepDriveServerProxy methods
	*/

	virtual void RegisterClient(int32 ClientId, bool IsMaster);

	virtual void UnregisterClient(int32 ClientId, bool IsMaster);

	virtual int32 RegisterCaptureCamera(float FieldOfView, int32 CaptureWidth, int32 CaptureHeight, FVector RelativePosition, FVector RelativeRotation, const FString &Label);

	virtual bool RequestAgentControl();

	virtual void ReleaseAgentControl();

	virtual void ResetAgent();

	virtual void SetAgentControlValues(float steering, float throttle, float brake, bool handbrake);


	UFUNCTION(BlueprintCallable, Category = "Input")
	void MoveForward(float AxisValue);

	UFUNCTION(BlueprintCallable, Category = "Input")
	void MoveRight(float AxisValue);

	UFUNCTION(BlueprintCallable, Category = "Input")
	void LookUp(float AxisValue);

	UFUNCTION(BlueprintCallable, Category = "Input")
	void Turn(float AxisValue);

	UFUNCTION(BlueprintCallable, Category = "Input")
	void OnCameraSelect(EDeepDriveAgentCameraType CameraType);

	UFUNCTION(BlueprintCallable, Category = "Input")
	void OnSelectMode(EDeepDriveAgentControlMode Mode);
	

private:

	EDeepDriveAgentCameraType				m_curCameraType;

};
