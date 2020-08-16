

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "ActorEventLoggingManager.generated.h"

class UActorEventLoggerComponent;

UCLASS()
class ACTOREVENTLOGGING_API AActorEventLoggingManager : public AActor
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	AActorEventLoggingManager()
    {   }

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override
    {   }

	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override
    {   }

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override
    {   }

	void RegisterActorEventLogger(UActorEventLoggerComponent& ActorEventLogger)
    {   }

};
