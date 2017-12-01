// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "Camera/CameraComponent.h"
#include "Public/Capture/CaptureDefines.h"
#include "CaptureCameraComponent.generated.h"

DECLARE_LOG_CATEGORY_EXTERN(DeepDriveCaptureComponent, Log, All);

struct SCaptureRequest;

/**
 * 
 */
UCLASS(meta=(BlueprintSpawnableComponent), Category = "DeepDrivePlugin")
class DEEPDRIVEPLUGIN_API UCaptureCameraComponent : public UCameraComponent
{
	GENERATED_BODY()
	
public:

	UCaptureCameraComponent();

	UPROPERTY(EditAnywhere, Category = "CaptureCamera")
	EDeepDriveCameraType	CameraType = EDeepDriveCameraType::DDC_CAMERA_NONE; 

	UPROPERTY(BlueprintReadOnly, Category = "CaptureCamera")
	int32	CameraId; 

	UPROPERTY(EditDefaultsOnly, BlueprintReadOnly, Category = "CaptureCamera")
	bool	IsCapturingActive = true;

	UPROPERTY(EditDefaultsOnly, Category = "CaptureCamera")
	bool	CaptureSceneEveryFrame = false;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "CaptureCamera")
	UTextureRenderTarget2D	*SceneRenderTarget;

	UFUNCTION(BlueprintCallable, Category = "Capturing")
	void Initialize(UTextureRenderTarget2D *RenderTarget);


	UFUNCTION(BlueprintCallable, Category="Capturing")
	void ActivateCapturing();

	UFUNCTION(BlueprintCallable, Category="Capturing")
	void DeactivateCapturing();

	bool capture(SCaptureRequest &reqData);

private:

	UPROPERTY()
	USceneCaptureComponent2D	*m_SceneCapture;
};
