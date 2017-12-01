// Fill out your copyright notice in the Description page of Project Settings.

#include "DeepDrivePluginPrivatePCH.h"
#include "DeepDriveServerProxy.h"
#include "Private/Server/DeepDriveServer.h"

DEFINE_LOG_CATEGORY(LogDeepDriveServerProxy);


// Sets default values
ADeepDriveServerProxy::ADeepDriveServerProxy()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

}

void ADeepDriveServerProxy::PreInitializeComponents()
{
	Super::PreInitializeComponents();

	bool alreadyRegistered = false;
	TArray<AActor*> proxies;
	UGameplayStatics::GetAllActorsOfClass(GetWorld(), TSubclassOf<ADeepDriveServerProxy>(), proxies );
	for(auto &actor : proxies)
	{
		ADeepDriveServerProxy *proxy = Cast<ADeepDriveServerProxy> (actor);
		if	(	proxy
			&&	proxy != this
			)
		{
			if(proxy->m_isActive)
			{
				alreadyRegistered = true;
				UE_LOG(LogDeepDriveServerProxy, Log, TEXT("Another Server Proxy [%s] is already registered"), *(proxy->GetFullName()));
				break;
			}
		}
	}

	if(!alreadyRegistered)
	{
		if(DeepDriveServer::GetInstance().RegisterProxy(*this))
		{
			m_isActive = true;
			UE_LOG(LogDeepDriveServerProxy, Log, TEXT("Server Proxy [%s] registered"), *(GetFullName()));
		}
	}

}

// Called when the game starts or when spawned
void ADeepDriveServerProxy::BeginPlay()
{
	Super::BeginPlay();
	
}

void ADeepDriveServerProxy::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	if (m_isActive)
	{
		DeepDriveServer::GetInstance().UnregisterProxy(*this);
		m_isActive = false;
		UE_LOG(LogDeepDriveServerProxy, Log, TEXT("Server Proxy [%s] unregistered"), *(GetFullName()));
	}
}

// Called every frame
void ADeepDriveServerProxy::Tick( float DeltaTime )
{
	Super::Tick( DeltaTime );

	if (m_isActive)
	{
		DeepDriveServer::GetInstance().update(DeltaTime);
	}

}
