// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "ActorEventLoggingReplayComponent.generated.h"


UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent) )
class ACTOREVENTLOGGING_API UActorEventLoggingReplayComponent : public UActorComponent
{
	GENERATED_BODY()

public:	

	UFUNCTION(BlueprintNativeEvent, Category = "Replay")
	void SetActorTransform(const FTransform &Transform);

};

inline void UActorEventLoggingReplayComponent::SetActorTransform_Implementation(const FTransform &Transform)
{
}
