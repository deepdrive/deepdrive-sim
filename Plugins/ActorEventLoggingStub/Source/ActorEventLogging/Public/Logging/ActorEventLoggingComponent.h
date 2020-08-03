

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"

#include <fstream>

#include "ActorEventLoggingComponent.generated.h"


DECLARE_LOG_CATEGORY_EXTERN(LogActorEventLogger, Log, All);

struct ActorEvent;
class AActorEventLoggingManager;

UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent) )
class ACTOREVENTLOGGING_API UActorEventLoggingComponent : public UActorComponent
{
	GENERATED_BODY()

public:	
	// Sets default values for this component's properties
	UActorEventLoggingComponent()
	{	}

protected:

public:	

	void StartLogging(const FString &BasePath)
	{	}

	void StopLogging()
	{	}

	void BeginFrame(uint32 FrameCounter, double Timestamp)
	{	}

	void EndFrame(double Timestamp)
	{	}

	UFUNCTION(BlueprintCallable, Category = "Event Logging")
	void LogMessage(const FString &Message)
	{	}

	UFUNCTION(BlueprintCallable, Category = "Event Logging")
	void LogActorTransform(const FTransform &Transform)
	{	}

	void LogEvent(const ActorEvent &Event)
	{	}

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Configuration)
	FString		ActorName = "";

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Configuration)
	FString		ActorDescription = "";

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Configuration)
	FString		ActorClassName = "";

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Configuration)
	FString		LogFileName = "";

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Configuration)
	bool 	KeepTransformFixed = false;


private:
};

