
#pragma once

#include "ActorEventLogging/ActorEventLoggerComponent.h"
#include "ActorEventLogging/ActorEventLogManager.h"

#define AEL_MESSAGE(ActorRef, Format, ...) \
{ \
	UActorEventLoggerComponent *eventLogCmp = Cast<UActorEventLoggerComponent>(ActorRef.GetComponentByClass(UActorEventLoggerComponent::StaticClass())); \
	if(eventLogCmp) \
		eventLogCmp->LogMessage(FString::Printf(Format, ##__VA_ARGS__)); \
}

#define AEL_ENSURE_TICK_ORDER(ActorRef) \
{ \
   	TArray<AActor*> actors; \
	UGameplayStatics::GetAllActorsOfClass(GetWorld(), AActorEventLogManager::StaticClass(), actors); \
	for (auto &actor : actors) \
	{ \
		AActorEventLogManager *mgr = Cast<AActorEventLogManager>(actor); \
		if (mgr) \
		{ \
			(ActorRef).AddTickPrerequisiteActor(mgr); \
            break; \
		} \
	} \
}
