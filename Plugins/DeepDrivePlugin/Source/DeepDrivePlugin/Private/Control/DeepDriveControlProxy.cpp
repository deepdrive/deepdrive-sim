// Fill out your copyright notice in the Description page of Project Settings.

#include "DeepDrivePluginPrivatePCH.h"
#include "DeepDriveControlProxy.h"

#include "Public/Messages/DeepDriveControlMessages.h"

// Sets default values
ADeepDriveControlProxy::ADeepDriveControlProxy()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

}

void ADeepDriveControlProxy::PreInitializeComponents()
{
	Super::PreInitializeComponents();
}


// Called when the game starts or when spawned
void ADeepDriveControlProxy::BeginPlay()
{
	Super::BeginPlay();
	
	bool alreadyRegistered = false;
	TArray<AActor*> proxies;
	UGameplayStatics::GetAllActorsOfClass(GetWorld(), TSubclassOf<ADeepDriveControlProxy>(), proxies);
	for (auto &actor : proxies)
	{
		ADeepDriveControlProxy *proxy = Cast<ADeepDriveControlProxy>(actor);
		if (proxy
			&&	proxy != this
			)
		{
			if (proxy->m_isActive)
			{
				alreadyRegistered = true;
				UE_LOG(LogDeepDriveControl, Log, TEXT("Another Control Proxy [%s] is already registered"), *(proxy->GetFullName()));
				break;
			}
		}
	}

	if (!alreadyRegistered)
	{
		const FString &sharedMemName = UGameplayStatics::GetPlatformName() == "Linux" ? SharedMemNameLinux : SharedMemNameWindows;
		DeepDriveControl::GetInstance().RegisterProxy(*this, sharedMemName, MaxSharedMemSize);
		m_isActive = true;
		UE_LOG(LogDeepDriveControl, Log, TEXT("Control Proxy [%s] registered"), *(GetFullName()));
	}
}

void ADeepDriveControlProxy::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	if (m_isActive)
	{
		DeepDriveControl::GetInstance().UnregisterProxy(*this);
		m_isActive = false;
		UE_LOG(LogDeepDriveControl, Log, TEXT("Control Proxy [%s] unregistered"), *(GetFullName()));
	}
}


// Called every frame
void ADeepDriveControlProxy::Tick( float DeltaTime )
{
	Super::Tick( DeltaTime );

	if (m_isActive)
	{
		const DeepDriveMessageHeader *msg = DeepDriveControl::GetInstance().getMessage();

		if (msg)
		{
			switch (msg->message_type)
			{
				case DeepDriveMessageType::Control:
					{
						const DeepDriveControlMessage *ctrlMsg = static_cast<const DeepDriveControlMessage*> (msg);
						FDeepDriveControlData ctrlData;
						ctrlData.Steering = ctrlMsg->steering;
						ctrlData.Throttle = ctrlMsg->throttle;
						ctrlData.Brake = ctrlMsg->brake;
						ctrlData.Handbrake = ctrlMsg->handbrake;
						ctrlData.IsGameDriving = ctrlMsg->is_game_driving > 0 ? true : false;
						ctrlData.ShouldReset = ctrlMsg->should_reset > 0 ? true : false;

						const double curTS = FPlatformTime::Seconds();
						const float roundTrip = static_cast<float> ((curTS - ctrlMsg->capture_timestamp) * 1000.0);

						UE_LOG(LogDeepDriveControl, VeryVerbose, TEXT("Control Proxy Round trip time: %f msecs"), roundTrip );

						OnNewControlData(ctrlData);
					}
					break;
			}

			FMemory::Free(const_cast<DeepDriveMessageHeader*> (msg));
		}

	}

}

