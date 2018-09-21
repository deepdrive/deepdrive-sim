// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "Camera/CameraComponent.h"
#include "Public/Capture/CaptureDefines.h"
#include "CaptureCameraComponent.generated.h"

DECLARE_LOG_CATEGORY_EXTERN(DeepDriveCaptureComponent, Log, All);

struct SCaptureRequest;
struct FDeepDriveViewMode;

/**
 * 
 */
UCLASS(meta=(BlueprintSpawnableComponent), Category = "DeepDrivePlugin")
class DEEPDRIVEPLUGIN_API UCaptureCameraComponent : public UCameraComponent
{
	GENERATED_BODY()
	
public:

	UCaptureCameraComponent();

	virtual void OnUnregister() override;

	UPROPERTY(EditAnywhere, Category = "CaptureCamera")
	EDeepDriveCameraType	CameraType = EDeepDriveCameraType::DDC_CAMERA_NONE; 

	UPROPERTY(BlueprintReadOnly, Category = "CaptureCamera")
	int32	CameraId; 

	UPROPERTY(EditDefaultsOnly, BlueprintReadOnly, Category = "CaptureCamera")
	bool	IsCapturingActive = true;

	UPROPERTY(EditDefaultsOnly, Category = "CaptureCamera")
	bool	CaptureSceneEveryFrame = false;

	UFUNCTION(BlueprintCallable, Category = "CaptureCamera")
	void Initialize(UTextureRenderTarget2D *colorRenderTarget, UTextureRenderTarget2D *depthRenderTarget, float FoV);

	UFUNCTION(BlueprintCallable, Category = "CaptureCamera")
	void Remove();

	UFUNCTION(BlueprintCallable, Category="Capturing")
	void ActivateCapturing();

	UFUNCTION(BlueprintCallable, Category="Capturing")
	void DeactivateCapturing();

	bool capture(SCaptureRequest &reqData);

	int32 getCameraId() const;

	void setViewMode(const FDeepDriveViewMode *viewMode);

	UTextureRenderTarget2D* getDepthRenderTexture();

private:

	UPROPERTY()
	USceneCaptureComponent2D			*m_SceneCapture = 0;

	UPROPERTY()
	USceneCaptureComponent2D			*m_DepthCapture = 0;

	UPROPERTY()
	UTextureRenderTarget2D 				*m_DepthRenderTexture = 0;

	bool								m_hasValidViewMode = false;
	EDeepDriveInternalCaptureEncoding	m_InternalCaptureEncoding = EDeepDriveInternalCaptureEncoding::RGB_DEPTH;
	bool								m_needsSeparateDepthCapture;
};

inline int32 UCaptureCameraComponent::getCameraId() const
{
	return CameraId;
}

inline UTextureRenderTarget2D* UCaptureCameraComponent::getDepthRenderTexture()
{
	return m_DepthRenderTexture;
}
