// Fill out your copyright notice in the Description page of Project Settings.

#include "DeepDrivePluginPrivatePCH.h"
#include "DeepDrivePlugin.h"
#include "Public/Capture/DeepDriveCaptureProxy.h"

#include "Private/Capture/DeepDriveCapture.h"

#include "Public/CaptureSink/CaptureSinkComponentBase.h"

DEFINE_LOG_CATEGORY(DeepDriveCaptureProxy);

// Sets default values
ADeepDriveCaptureProxy::ADeepDriveCaptureProxy()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

}

void ADeepDriveCaptureProxy::PreInitializeComponents()
{
	Super::PreInitializeComponents();

	bool alreadyRegistered = false;
	TArray<AActor*> proxies;
	UGameplayStatics::GetAllActorsOfClass(GetWorld(), TSubclassOf<ADeepDriveCaptureProxy>(), proxies );
	for(auto &actor : proxies)
	{
		ADeepDriveCaptureProxy *proxy = Cast<ADeepDriveCaptureProxy> (actor);
		if	(	proxy
			&&	proxy != this
			)
		{
			if(proxy->m_isActive)
			{
				alreadyRegistered = true;
				UE_LOG(LogDeepDriveCapture, Log, TEXT("Another Capture Proxy [%s] is already registered"), *(proxy->GetFullName()));
				break;
			}
		}
	}

	if(!alreadyRegistered)
	{
		DeepDriveCapture::GetInstance().RegisterProxy(*this);
		m_isActive = true;
		UE_LOG(LogDeepDriveCapture, Log, TEXT("Capture Proxy [%s] registered"), *(GetFullName()));
	}
}


// Called when the game starts or when spawned
void ADeepDriveCaptureProxy::BeginPlay()
{
	Super::BeginPlay();

	m_TimeToNextCapture = CaptureInterval;

	if(m_isActive)
	{
		const TSet < UActorComponent * > &components = GetComponents();
		for(auto &comp : components)
		{
			UCaptureSinkComponentBase *captureSinkComp = Cast<UCaptureSinkComponentBase> (comp);
			if(captureSinkComp)
			{
				m_CaptureSinks.Add(captureSinkComp);
				UE_LOG(LogDeepDriveCapture, Log, TEXT("Found sink %s"), *(captureSinkComp->getName()));
			}
		}
	}
	
}

void ADeepDriveCaptureProxy::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	Super::EndPlay(EndPlayReason);

	if(m_isActive)
	{
		DeepDriveCapture::GetInstance().UnregisterProxy(*this);
		UE_LOG(LogDeepDriveCapture, Log, TEXT("Proxy unregistered"));
		m_isActive = false;
	}

}

// Called every frame
void ADeepDriveCaptureProxy::Tick( float DeltaTime )
{
	Super::Tick( DeltaTime );

	if(m_isActive)
	{
		DeepDriveCapture &deepDriveCapture = DeepDriveCapture::GetInstance();

		deepDriveCapture.HandleCaptureResult();

		if(CaptureInterval >= 0.0f)
		{
			m_TimeToNextCapture -= DeltaTime;

			if(m_TimeToNextCapture <= 0.0f)
			{
				m_DeepDriveData = BeginCapture();
				DeepDriveCapture::GetInstance().Capture();

				m_TimeToNextCapture = CaptureInterval;
			}
		}
	}
}


void ADeepDriveCaptureProxy::Capture()
{
	UE_LOG(LogDeepDriveCapture, Log, TEXT("ADeepDriveCaptureProxy::Capture isActive %c"), m_isActive ? 'T' : 'F');
	if(m_isActive)
		DeepDriveCapture::GetInstance().Capture();
}
