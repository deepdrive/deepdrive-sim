

#include "DeepDrivePluginPrivatePCH.h"
#include "ActorEventLogEndFrameComponent.h"
#include "ActorEventLogManager.h"

// Sets default values for this component's properties
UActorEventLogEndFrameComponent::UActorEventLogEndFrameComponent()
{
	// Set this component to be initialized when the game starts, and to be ticked every frame.  You can turn these features
	// off to improve performance if you don't need them.
	PrimaryComponentTick.bCanEverTick = true;

	SetTickGroup(TG_PostUpdateWork);
}


// Called when the game starts
void UActorEventLogEndFrameComponent::BeginPlay()
{
	Super::BeginPlay();
	SetTickGroup(TG_PostUpdateWork);
}


// Called every frame
void UActorEventLogEndFrameComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
	if (m_ActorEventLogMgr)
		m_ActorEventLogMgr->OnEndFrame();
}

