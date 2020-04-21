

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"

#include <fstream>

#include "ActorEventLoggerComponent.generated.h"


DECLARE_LOG_CATEGORY_EXTERN(LogActorEventLogger, Log, All);

struct ActorEvent;
class AActorEventLogManager;

UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent) )
class DEEPDRIVEPLUGIN_API UActorEventLoggerComponent : public UActorComponent
{
	GENERATED_BODY()

public:	
	// Sets default values for this component's properties
	UActorEventLoggerComponent();

protected:
	// Called when the game starts
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

	void StartLogging(const FString &BasePath);

	void StopLogging();

	void BeginFrame(uint32 FrameCounter, double Timestamp);

	void EndFrame(double Timestamp);

	void LogMessage(const FString &Message);

	void LogActorTransform(const FTransform &Transform);

	void LogEvent(const ActorEvent &Event);

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Configuration)
	FString		ActorName = "";

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Configuration)
	FString		ActorDescription = "";

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Configuration)
	FString		ActorClassName = "";

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Configuration)
	bool 	KeepTransformFixed = false;

	void setUniqueActorName(const FString &name);

private:

	AActorEventLogManager			*m_ActorEventLogMgr = 0;

	FString							m_EventLogFileName;
	FString							m_UniqueActorName;

	std::ofstream					m_LogStream;

	bool							m_isFirstFrame = true;
	bool							m_isFirstEvent = true;
	bool							m_hasSeenBeginFrame = false;
};


inline void UActorEventLoggerComponent::setUniqueActorName(const FString &name)
{
	m_UniqueActorName = name;
}
