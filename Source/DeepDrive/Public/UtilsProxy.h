

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Runtime/Landscape/Classes/LandscapeSplineSegment.h"

#include "UtilsProxy.generated.h"

//DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveUtilsProxy, Log, All);

UCLASS()
class DEEPDRIVE_API AUtilsProxy : public AActor
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	AUtilsProxy();

	UFUNCTION(BlueprintCallable, Category = "Python")
	int32 GetSplineSegmentConnections(UObject* StructProperty);

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

	
	
};
