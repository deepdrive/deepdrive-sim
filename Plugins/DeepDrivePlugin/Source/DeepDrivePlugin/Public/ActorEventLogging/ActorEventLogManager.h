

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "ActorEventLogManager.generated.h"

class UActorEventLoggerComponent;
class UActorEventLogEndFrameComponent;

UCLASS()
class DEEPDRIVEPLUGIN_API AActorEventLogManager : public AActor
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	AActorEventLogManager();

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

	void RegisterActorEventLogger(UActorEventLoggerComponent& ActorEventLogger);

	void OnEndFrame();

	double getTimestamp() const;

private:

	UPROPERTY()
	UActorEventLogEndFrameComponent				*m_EndFrameCmp = 0;

	TArray<UActorEventLoggerComponent*>			m_ActorEventLoggers;

	uint32										m_FrameCounter = 0;
	double										m_StartTimestamp = 0.0;

};


inline double AActorEventLogManager::getTimestamp() const
{
	return FPlatformTime::Seconds() - m_StartTimestamp;
}