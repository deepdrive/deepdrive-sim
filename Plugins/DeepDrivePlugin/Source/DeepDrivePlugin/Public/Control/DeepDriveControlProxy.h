// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "GameFramework/Actor.h"
#include "DeepDriveData.h"
#include "DeepDriveControlProxy.generated.h"

UCLASS()
class DEEPDRIVEPLUGIN_API ADeepDriveControlProxy : public AActor
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	ADeepDriveControlProxy();

	virtual void PreInitializeComponents() override;

	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

	// Called when the game starts or when spawned
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

	// Called every frame
	virtual void Tick( float DeltaSeconds ) override;

	UFUNCTION(BlueprintImplementableEvent, Category = "DeepDriveControl")
	void OnNewControlData(const FDeepDriveControlData &controlData);

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = SharedMem)
	FString		SharedMemNameLinux;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = SharedMem)
	FString		SharedMemNameWindows;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = SharedMem)
	int32 MaxSharedMemSize = 1 * 1024 * 1024;
	

private:

	bool					m_isActive = false;
};
