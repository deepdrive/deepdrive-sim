

#include "DeepDrivePluginPrivatePCH.h"
#include "ActorEventLogReplayManager.h"
#include "ReplayActorEventLog.h"

DEFINE_LOG_CATEGORY(LogActorEventReplayManager);

// Sets default values
AActorEventLogReplayManager::AActorEventLogReplayManager()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

}

// Called when the game starts or when spawned
void AActorEventLogReplayManager::BeginPlay()
{
	Super::BeginPlay();
	
}

// Called every frame
void AActorEventLogReplayManager::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	switch(m_curState)
	{
		case Idle:
			for (auto &ael : m_ActorEventLogs)
				ael.Value->updateOnIdle(DeltaTime);
			break;

		case Loading:
			break;

		case Replaying:
			++m_curFrameIndex;
			if(m_curFrameIndex < m_FrameCount)
			{
				replayCurrentFrame();
				propagateFrameIndex();
			}
			else
				m_curFrameIndex = m_FrameCount - 1;
			break;
	}

}

FActorEventLogData AActorEventLogReplayManager::LoadActorEventLog(FString FileName)
{
	FActorEventLogData aelData;
	
	bool alreadyLoaded = false;
	for(auto &a : m_ActorEventLogs)
	{
		if(a.Value->getFileName() == FileName)
		{
			alreadyLoaded = true;
			break;
		}
	}

	if(alreadyLoaded == false)
	{
		const int32 id = m_ActorEventLogs.Num() + 1;

		ReplayActorEventLog *rael = new ReplayActorEventLog(*this, id);
		if	(	rael
			&&	rael->load(FileName)
			)
		{
			const FTransform &transform = rael->getInitialTransform();
			const FString &className = rael->getActorClassName();
			AActor* actor = className.IsEmpty() ? spawnActor(transform.GetLocation(), transform.GetRotation().Rotator()) : spawnActor(className, transform.GetLocation(), transform.GetRotation().Rotator());
			if(actor)
			{
				rael->setActor(actor);
				if (DisablePhysics)
					rael->setPhysicsEnabled(false, false);

				m_ActorEventLogs.Add(id, rael);

				aelData.Id = id;
				aelData.ActorName = rael->getActorName();
				aelData.ActorDescription = rael->getActorDescription();

				updateFrameCount();
				rael->replayFrame(m_curFrameIndex);
			}
			else
				UE_LOG(LogActorEventReplayManager, Log, TEXT("Spawning actor failed") );
		}
	}
	else
		UE_LOG(LogActorEventReplayManager, Log, TEXT("Actor event logfile %s already loaded"), *(FileName));

	return aelData;
}

TArray<FString> AActorEventLogReplayManager::GetAvailableActorEventLogs(FString BasePath)
{
	TArray<FString> logs;

	IFileManager& fileMgr = IFileManager::Get();
	FString finalPath = BasePath + "/*.ael";
	fileMgr.FindFiles(logs, *finalPath, true, false);

	return logs;
}

void AActorEventLogReplayManager::GoToFrame(int32 FrameIndex)
{
	if(m_curState == Idle)
	{
		m_curFrameIndex = FrameIndex;
		if	(	m_curFrameIndex < 0
			||	m_curFrameIndex >= m_FrameCount
			)
			m_curFrameIndex = m_FrameCount - 1;
		
		replayCurrentFrame();
		propagateFrameIndex();
	}
}

void AActorEventLogReplayManager::GoToFirstFrame()
{
	if (m_curState == Idle)
	{
		m_curFrameIndex = 0;
		replayCurrentFrame();
		propagateFrameIndex();
	}
}

void AActorEventLogReplayManager::GoToLastFrame()
{
	if (m_curState == Idle)
	{
		m_curFrameIndex = m_FrameCount - 1;
		replayCurrentFrame();
		propagateFrameIndex();
	}
}

void AActorEventLogReplayManager::NextFrame()
{
	if(m_curState == Idle)
	{
		++m_curFrameIndex;
		if(m_curFrameIndex >= m_FrameCount)
			m_curFrameIndex = m_FrameCount - 1;
		
		replayCurrentFrame();
		propagateFrameIndex();
	}
}

void AActorEventLogReplayManager::PreviousFrame()
{
	if(m_curState == Idle)
	{
		--m_curFrameIndex;
		if(m_curFrameIndex < 0)
			m_curFrameIndex = 0;
		
		replayCurrentFrame();
		propagateFrameIndex();
	}
}

void AActorEventLogReplayManager::Replay()
{
	if(m_ActorEventLogs.Num() > 0)
		m_curState = Replaying;
}

void AActorEventLogReplayManager::StopReplay()
{
	m_curState = Idle;
}

void AActorEventLogReplayManager::TogglePause()
{
	m_curState = Idle;
}

void AActorEventLogReplayManager::replayCurrentFrame()
{
	if (OnBeginReplayFrame.IsBound())
	{
		OnBeginReplayFrame.Broadcast();
	}

	for (auto &ael : m_ActorEventLogs)
		ael.Value->replayFrame(m_curFrameIndex);
}

AActor* AActorEventLogReplayManager::spawnActor(const FString &className, const FVector &location, const FRotator &rotation)
{
	AActor *actor = 0;
	UObject* actorObj = Cast<UObject>(StaticLoadObject(UObject::StaticClass(), NULL, *className));

	if (actorObj)
	{
		UBlueprint* bpObj = Cast<UBlueprint>(actorObj);
		UClass* classToSpawn = actorObj->StaticClass();
		if (classToSpawn)
		{
			UWorld* World = GetWorld();
			FActorSpawnParameters SpawnParams;
			SpawnParams.Owner = this;
			SpawnParams.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
			actor = World->SpawnActor<AActor>(bpObj->GeneratedClass, location, rotation, SpawnParams);
		}
		else
		{
			GEngine->AddOnScreenDebugMessage(-1, 1.f, FColor::Red, FString::Printf(TEXT("CLASS == NULL")));
		}
	}
	else
	{
		GEngine->AddOnScreenDebugMessage(-1, 1.f, FColor::Red, FString::Printf(TEXT("CANT FIND OBJECT TO SPAWN")));
	}

	return actor;
}

AActor* AActorEventLogReplayManager::spawnActor( const FVector &location, const FRotator &rotation)
{
	FActorSpawnParameters SpawnParams;
	SpawnParams.Owner = this;
	SpawnParams.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
	return GetWorld()->SpawnActor<AActor>(DummyActor, location, rotation, SpawnParams);
}

void AActorEventLogReplayManager::updateFrameCount()
{
	int32 frameCount = 0;
	for(auto &ael : m_ActorEventLogs)
	{
		const int32 curCount = ael.Value->getFrameCount();
		if (curCount > frameCount)
			frameCount = curCount;
	}

	if(frameCount != m_FrameCount)
	{
		m_FrameCount = frameCount;
		if (OnFrameCountChanged.IsBound())
		{
			OnFrameCountChanged.Broadcast(frameCount);
		}

	}
}

void AActorEventLogReplayManager::propagateFrameIndex()
{
	if (OnFrameIndexChanged.IsBound())
	{
		OnFrameIndexChanged.Broadcast(m_curFrameIndex);
	}
}
