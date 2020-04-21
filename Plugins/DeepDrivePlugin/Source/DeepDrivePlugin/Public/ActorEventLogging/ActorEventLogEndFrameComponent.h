
#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "ActorEventLogEndFrameComponent.generated.h"

class AActorEventLogManager;

UCLASS( ClassGroup=(Custom) )
class DEEPDRIVEPLUGIN_API UActorEventLogEndFrameComponent : public UActorComponent
{
	GENERATED_BODY()

public:	
	// Sets default values for this component's properties
	UActorEventLogEndFrameComponent();

protected:
	// Called when the game starts
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

	void setActorEventLogManager(AActorEventLogManager &mgr);

private:

	AActorEventLogManager			*m_ActorEventLogMgr = 0;
};


inline void UActorEventLogEndFrameComponent::setActorEventLogManager(AActorEventLogManager &mgr)
{
	m_ActorEventLogMgr = &mgr;
}
