// Fill out your copyright notice in the Description page of Project Settings.

#include "DeepDrivePluginPrivatePCH.h"
#include "Public/Capture/CaptureCameraComponent.h"
#include "Private/Capture/DeepDriveCapture.h"
#include "Public/Capture/CaptureDefines.h"
#include "Private/Capture/CaptureJob.h"


DEFINE_LOG_CATEGORY(DeepDriveCaptureComponent);

UCaptureCameraComponent::UCaptureCameraComponent()
	:	m_SceneCapture(0)
{
	bWantsInitializeComponent = true;
}

void UCaptureCameraComponent::Initialize(UTextureRenderTarget2D *RenderTarget)
{
	CameraId = DeepDriveCapture::GetInstance().RegisterCaptureComponent(this);

	AActor *owningActor = GetOwner();
	FString compName = "SceneCaptureComponent_" + FString::FromInt(CameraId);

	m_SceneCapture = NewObject<USceneCaptureComponent2D>(owningActor, FName(*compName));
	m_SceneCapture->RegisterComponent();

	UE_LOG(DeepDriveCaptureComponent, Log, TEXT("Scene capture setup for %s at %p"), *(compName), m_SceneCapture);

	if (m_SceneCapture)
	{
		m_SceneCapture->TextureTarget = RenderTarget;			
		m_SceneCapture->CaptureSource = ESceneCaptureSource::SCS_SceneColorSceneDepth;

		m_SceneCapture->HiddenActors.Add(GetOwner());

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
	}

	UE_LOG(DeepDriveCaptureComponent, Log, TEXT("UCaptureCameraComponent::InitializeComponent 0x%p camId %d"), m_SceneCapture, CameraId);
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
			if(!CaptureSceneEveryFrame)
				m_SceneCapture->CaptureScene();
			reqData.capture_source = sceneSrc;
			reqData.camera_type = CameraType;
			reqData.camera_id = CameraId;
			shallCapture = true;
		}
	}

	return shallCapture;
}
