// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "GameFramework/Actor.h"
#include "DeepDriveData.h"
#include "CaptureDefines.h"
#include "DeepDriveCaptureProxy.generated.h"

DECLARE_LOG_CATEGORY_EXTERN(DeepDriveCaptureProxy, Log, All);

class UCaptureSinkComponentBase;


USTRUCT(BlueprintType)
struct FCaptureCyle
{
	GENERATED_USTRUCT_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	TArray<EDeepDriveCameraType>	Cameras;
};

UCLASS()
class DEEPDRIVEPLUGIN_API ADeepDriveCaptureProxy : public AActor
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	ADeepDriveCaptureProxy();

	// Called when the game starts or when spawned
	virtual void PreInitializeComponents() override;

	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

	// Called when the game ends or actor is removed from level
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
	
	// Called every frame
	virtual void Tick( float DeltaSeconds ) override;

	UFUNCTION(BlueprintImplementableEvent, Category="DeepDrivePlugin")
	FDeepDriveDataOut BeginCapture();


	UFUNCTION(BlueprintCallable, Category="DeepDrivePlugin")
	void Capture();

	TArray<UCaptureSinkComponentBase*>& getSinks();

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Capturing)
	float	CaptureInterval = 0.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Capturing)
	TArray< FCaptureCyle >	CaptureCycles;

	const FDeepDriveDataOut& getDeepDriveData() const;

private:

	bool									m_isActive = false;

	TArray<UCaptureSinkComponentBase*>		m_CaptureSinks;
	
	float									m_TimeToNextCapture;

	FDeepDriveDataOut						m_DeepDriveData;

};


inline TArray<UCaptureSinkComponentBase*>& ADeepDriveCaptureProxy::getSinks()
{
	return m_CaptureSinks;
}

inline const FDeepDriveDataOut& ADeepDriveCaptureProxy::getDeepDriveData() const
{
	return m_DeepDriveData;
}
