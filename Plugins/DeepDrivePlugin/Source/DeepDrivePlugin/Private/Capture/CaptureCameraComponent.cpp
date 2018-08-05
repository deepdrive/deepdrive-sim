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

	ConstructorHelpers::FObjectFinder<UMaterialInstance> Material(TEXT("Material'/Game/DeepDrive/Materials/M_PostProcess_Inst.M_PostProcess_Inst'"));

	m_PostProcessMat = Material.Succeeded() ? Material.Object : 0;

	UE_LOG(DeepDriveCaptureComponent, Log, TEXT("PostProcessMat ==> Found %c ptr %p"), Material.Succeeded() ? 'T' : 'F', m_PostProcessMat);
}

void UCaptureCameraComponent::Initialize(UTextureRenderTarget2D *RenderTarget, float FoV)
{
	CameraId = DeepDriveCapture::GetInstance().RegisterCaptureComponent(this);

	AActor *owningActor = GetOwner();
	FString compName = "SceneCaptureComponent_" + FString::FromInt(CameraId);

	m_SceneCapture = NewObject<USceneCaptureComponent2D>(owningActor, FName(*compName));
	m_SceneCapture->RegisterComponent();

	UE_LOG(DeepDriveCaptureComponent, Log, TEXT("Scene capture setup for %s at %p"), *(compName), m_SceneCapture);

	if (m_SceneCapture)
	{
		SceneRenderTarget = RenderTarget;
		m_SceneCapture->TextureTarget = RenderTarget;			
		//m_SceneCapture->CaptureSource = ESceneCaptureSource::SCS_SceneColorSceneDepth;
		m_SceneCapture->CaptureSource = ESceneCaptureSource::SCS_FinalColorLDR;

		//m_SceneCapture->HiddenActors.Add(GetOwner());

		SceneCaptureCmp = m_SceneCapture;
		if (m_PostProcessMat && m_SceneCapture->PostProcessSettings.WeightedBlendables.Array.Num() == 0)
		{
			m_SceneCapture->PostProcessSettings.AddBlendable(m_PostProcessMat, 1.0f);
			UE_LOG(DeepDriveCaptureComponent, Log, TEXT("==> Adding PostProcessMat %d"), m_SceneCapture->PostProcessSettings.WeightedBlendables.Array.Num());
		}

		//m_SceneCapture->PostProcessSettings.bOverride_ColorSaturationMidtones = true;
		//m_SceneCapture->PostProcessSettings.ColorSaturationMidtones = FVector4(0.0f, 0.0f, 0.0f, 1.0f);

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
				m_SceneCapture->bCaptureEveryFrame = true;
				m_SceneCapture->CaptureScene();
				m_SceneCapture->bCaptureEveryFrame = false;
			}
			reqData.capture_source = sceneSrc;
			reqData.camera_type = CameraType;
			reqData.camera_id = CameraId;
			shallCapture = true;
		}
	}

	return shallCapture;
}
