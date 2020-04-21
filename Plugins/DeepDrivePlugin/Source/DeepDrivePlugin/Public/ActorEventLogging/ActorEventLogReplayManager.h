

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "ActorEventLogReplayManager.generated.h"

DECLARE_LOG_CATEGORY_EXTERN(LogActorEventReplayManager, Log, All);

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FFrameCountChanged, int32, FrameCount);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FFrameIndexChanged, int32, FrameIndex);

DECLARE_DYNAMIC_MULTICAST_DELEGATE(FBeginReplayFrame);

DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FMessageEvent, int32, Id, FString, Message);

USTRUCT(BlueprintType)
struct FActorEventLogData
{
	GENERATED_USTRUCT_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = Default)
	int32		Id = -1;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = Default)
	FString		ActorName;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = Default)
	FString		ActorDescription;

};

class ReplayActorEventLog;

UCLASS()
class DEEPDRIVEPLUGIN_API AActorEventLogReplayManager : public AActor
{
	GENERATED_BODY()

public:	
	// Sets default values for this actor's properties
	AActorEventLogReplayManager();

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

	UFUNCTION(BlueprintCallable, Category = "FileHandling")
	FActorEventLogData LoadActorEventLog(FString FileName);

	UFUNCTION(BlueprintCallable, Category = "FileHandling")
	TArray<FString> GetAvailableActorEventLogs(FString BasePath);

	UFUNCTION(BlueprintCallable, Category = "Control")
	void GoToFrame(int32 FrameIndex);

	UFUNCTION(BlueprintCallable, Category = "Control")
	void GoToFirstFrame();

	UFUNCTION(BlueprintCallable, Category = "Control")
	void GoToLastFrame();

	UFUNCTION(BlueprintCallable, Category = "Control")
	void NextFrame();

	UFUNCTION(BlueprintCallable, Category = "Control")
	void PreviousFrame();

	UFUNCTION(BlueprintCallable, Category = "Control")
	void Replay();

	UFUNCTION(BlueprintCallable, Category = "Control")
	void StopReplay();

	UFUNCTION(BlueprintCallable, Category = "Control")
	void TogglePause();

	UPROPERTY(BlueprintReadOnly, Category = Configuration)
	TSubclassOf<AActor>	DummyActor = 0;

	UPROPERTY(BlueprintReadOnly, Category = Configuration)
	bool DisablePhysics = true;

	UPROPERTY(BlueprintAssignable)
	FFrameCountChanged OnFrameCountChanged;

	UPROPERTY(BlueprintAssignable)
	FFrameIndexChanged OnFrameIndexChanged;

	UPROPERTY(BlueprintAssignable)
	FBeginReplayFrame OnBeginReplayFrame;

	UPROPERTY(BlueprintAssignable)
	FMessageEvent OnMessageEvent;

private:
	enum State
	{
		Idle,
		Loading,
		Replaying
	};

	void replayCurrentFrame();

	void updateFrameCount();

	void propagateFrameIndex();

	AActor* spawnActor(const FString &className, const FVector &location, const FRotator &rotation);
	AActor* spawnActor( const FVector &location, const FRotator &rotation);

	TMap<int32, ReplayActorEventLog*>			m_ActorEventLogs;

	State										m_curState = Idle;

	int32										m_FrameCount;
	int32										m_curFrameIndex;

};


