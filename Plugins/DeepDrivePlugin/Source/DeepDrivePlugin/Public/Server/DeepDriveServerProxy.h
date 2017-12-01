// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "GameFramework/Actor.h"
#include "Runtime/Sockets/Public/IPAddress.h"

#include "DeepDriveServerProxy.generated.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveServerProxy, Log, All);

UCLASS()
class DEEPDRIVEPLUGIN_API ADeepDriveServerProxy : public AActor
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	ADeepDriveServerProxy();

	// Called when the game starts or when spawned
	virtual void PreInitializeComponents() override;

	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

	// Called every frame
	virtual void Tick( float DeltaSeconds ) override;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Server)
	FString		IPAddress;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Server)
	int32		Port = 9876;

	UFUNCTION(BlueprintImplementableEvent, Category = "CameraConfiguration")
	int32 RegisterCamera(float FieldOfView, int32 CaptureWidth, int32 CaptureHeight, FVector RelativePosition, FVector RelativeRotation);

	UFUNCTION(BlueprintImplementableEvent, Category = "Control")
	bool RequestAgentControl();

	UFUNCTION(BlueprintImplementableEvent, Category = "Control")
	void ReleaseAgentControl();

	UFUNCTION(BlueprintImplementableEvent, Category = "Control")
	void SetAgentControlValues(float steering, float throttle, float brake, bool handbrake);

private:

	bool									m_isActive = false;

};
