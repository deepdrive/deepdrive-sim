// Fill out your copyright notice in the Description page of Project Settings.

#include "DeepDrivePluginPrivatePCH.h"
#include "Public/Capture/CaptureCameraComponent.h"
#include "Private/Capture/DeepDriveCapture.h"
#include "Public/Capture/CaptureDefines.h"
#include "Public/Simulation/DeepDriveSimulationDefines.h"
#include "Private/Capture/CaptureJob.h"


DEFINE_LOG_CATEGORY(DeepDriveCaptureComponent);

UCaptureCameraComponent::UCaptureCameraComponent()
	:	m_SceneCapture(0)
{
	bWantsInitializeComponent = true;
}

void UCaptureCameraComponent::Initialize(UTextureRenderTarget2D *colorRenderTarget, UTextureRenderTarget2D *depthRenderTarget, float FoV)
{
	CameraId = DeepDriveCapture::GetInstance().RegisterCaptureComponent(this);

	AActor *owningActor = GetOwner();

	m_SceneCapture = NewObject<USceneCaptureComponent2D>(owningActor, FName( *("ColorCaptureComponent_" + FString::FromInt(CameraId)) ));
	m_SceneCapture->RegisterComponent();

	m_DepthCapture = NewObject<USceneCaptureComponent2D>(owningActor, FName( *("DepthCaptureComponent_" + FString::FromInt(CameraId)) ));
	m_DepthCapture->RegisterComponent();

	if (m_SceneCapture && m_DepthCapture)
	{
		m_SceneCapture->TextureTarget = colorRenderTarget;
		m_SceneCapture->CaptureSource = ESceneCaptureSource::SCS_SceneColorSceneDepth;

		m_DepthCapture->TextureTarget = depthRenderTarget;
		m_DepthCapture->CaptureSource = ESceneCaptureSource::SCS_SceneDepth;


		//m_SceneCapture->HiddenActors.Add(GetOwner());


		if(IsCapturingActive)
		{
			m_SceneCapture->Activate();
			m_SceneCapture->bCaptureEveryFrame = CaptureSceneEveryFrame;
			m_SceneCapture->bCaptureOnMovement = CaptureSceneEveryFrame;
		}
		else
		{
			m_SceneCapture->Deactivate();
			m_SceneCapture->bCaptureEveryFrame = false;
			m_SceneCapture->bCaptureOnMovement = false;
		}

		m_SceneCapture->AttachToComponent(this, FAttachmentTransformRules(EAttachmentRule::KeepRelative, false));
		m_SceneCapture->FOVAngle = FoV;
	}

	UE_LOG(DeepDriveCaptureComponent, Log, TEXT("UCaptureCameraComponent::InitializeComponent 0x%p camId %d"), m_SceneCapture, CameraId);
}

void UCaptureCameraComponent::Remove()
{
	DeepDriveCapture::GetInstance().UnregisterCaptureComponent(CameraId);
	m_SceneCapture->DestroyComponent();
	//DestroyComponent();
}

void UCaptureCameraComponent::ActivateCapturing()
{
	if (m_SceneCapture)
	{
		m_SceneCapture->Activate();
		m_SceneCapture->bCaptureEveryFrame = CaptureSceneEveryFrame;
		m_SceneCapture->bCaptureOnMovement = CaptureSceneEveryFrame;
	}
	IsCapturingActive = true;
}


void UCaptureCameraComponent::DeactivateCapturing()
{
	if (m_SceneCapture)
	{
		m_SceneCapture->Deactivate();
		m_SceneCapture->bCaptureEveryFrame = false;
		m_SceneCapture->bCaptureOnMovement = false;
	}
	IsCapturingActive = false;
}

bool UCaptureCameraComponent::capture(SCaptureRequest &reqData)
{
	bool shallCapture = false;

	// UE_LOG(DeepDriveCaptureComponent, Log, TEXT("UCaptureCameraComponent::capture camId %d isActive %c"), CameraId, IsCapturingActive ? 'T' : 'F');

	if (IsCapturingActive)
	{
		FTextureRenderTargetResource *sceneSrc = m_SceneCapture ? m_SceneCapture->TextureTarget->GameThread_GetRenderTargetResource() : 0;
		if (sceneSrc)
		{
			if (!CaptureSceneEveryFrame)
			{
				if(m_hasValidViewMode)
					m_SceneCapture->bCaptureEveryFrame = true;

				m_SceneCapture->CaptureScene();

				if(m_hasValidViewMode)
					m_SceneCapture->bCaptureEveryFrame = false;
			}
			reqData.scene_capture_source = sceneSrc;

			FTextureRenderTargetResource *depthSrc = m_hasValidViewMode && m_DepthCapture ? m_DepthCapture->TextureTarget->GameThread_GetRenderTargetResource() : 0;
			if(depthSrc)
			{
				if (!CaptureSceneEveryFrame)
				{
					if(m_hasValidViewMode)
						m_DepthCapture->bCaptureEveryFrame = true;

					m_DepthCapture->CaptureScene();

					if(m_hasValidViewMode)
						m_DepthCapture->bCaptureEveryFrame = false;
				}
				reqData.depth_capture_source = depthSrc;
			}
			else
				reqData.depth_capture_source = 0;
	
			reqData.camera_type = CameraType;
			reqData.camera_id = CameraId;
			shallCapture = true;
		}
	}

	return shallCapture;
}

void UCaptureCameraComponent::setViewMode(const FDeepDriveViewMode *viewMode)
{
	if(viewMode)
	{
		m_SceneCapture->PostProcessSettings = FPostProcessSettings();

		if (viewMode->Material)
		{
			m_SceneCapture->PostProcessSettings.AddBlendable(viewMode->Material, 1.0f);
			m_hasValidViewMode = true;
		}

		if(m_hasValidViewMode)
			m_SceneCapture->CaptureSource = ESceneCaptureSource::SCS_FinalColorLDR;
	}
	else
	{
		m_SceneCapture->CaptureSource = ESceneCaptureSource::SCS_SceneColorSceneDepth;
	}
}
