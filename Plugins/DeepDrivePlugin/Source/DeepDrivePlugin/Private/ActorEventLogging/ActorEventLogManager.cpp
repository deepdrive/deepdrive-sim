

#include "DeepDrivePluginPrivatePCH.h"
#include "ActorEventLogManager.h"

#include "ActorEventLoggerComponent.h"
#include "ActorEventLogEndFrameComponent.h"


// Sets default values
AActorEventLogManager::AActorEventLogManager()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

	SetTickGroup(TG_PrePhysics);
	m_StartTimestamp = FPlatformTime::Seconds();

	m_EndFrameCmp = CreateDefaultSubobject<UActorEventLogEndFrameComponent>(TEXT("EndFrameCmp"));
	AddInstanceComponent(m_EndFrameCmp);
	m_EndFrameCmp->setActorEventLogManager(*this);
}

// Called when the game starts or when spawned
void AActorEventLogManager::BeginPlay()
{
	Super::BeginPlay();
	SetTickGroup(TG_PrePhysics);
	
}

// Called every frame
void AActorEventLogManager::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
	for(auto &l : m_ActorEventLoggers)
	{
		l->BeginFrame(m_FrameCounter, getTimestamp());
	}

	++m_FrameCounter;
}

void AActorEventLogManager::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	Super::EndPlay(EndPlayReason);

	for(auto &l : m_ActorEventLoggers)
	{
		l->StopLogging();
	}
}

void AActorEventLogManager::RegisterActorEventLogger(UActorEventLoggerComponent& ActorEventLogger)
{
	if(m_ActorEventLoggers.Contains(&ActorEventLogger) == false)
	{
		m_ActorEventLoggers.Add(&ActorEventLogger);

		ActorEventLogger.StartLogging(FPaths::ProjectLogDir());

		ActorEventLogger.GetOwner()->AddTickPrerequisiteActor(this);
	}
}

void AActorEventLogManager::OnEndFrame()
{
	for(auto &l : m_ActorEventLoggers)
	{
		l->EndFrame(getTimestamp());
	}
}
