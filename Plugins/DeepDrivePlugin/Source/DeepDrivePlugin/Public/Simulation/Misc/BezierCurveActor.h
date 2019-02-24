

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "BezierCurveActor.generated.h"

class UBezierCurveComponent;
class USphereComponent;

UCLASS()
class DEEPDRIVEPLUGIN_API ABezierCurveActor : public AActor
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	ABezierCurveActor();

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

	virtual bool ShouldTickIfViewportsOnly() const override;

	UFUNCTION(BlueprintCallable, Category = "Default")
	void ClearControlPoints();

	UFUNCTION(BlueprintCallable, Category = "Default")
	void AddControlPoint(USceneComponent *ControlPoint);

	UPROPERTY(EditAnywhere, Category = Debug)
	FColor						Color = FColor::Red;

protected:

	UPROPERTY(EditDefaultsOnly, Category = Default)
	USceneComponent				*Root = 0;

private:

	UPROPERTY(EditDefaultsOnly, Category = Default)
	UBezierCurveComponent		*m_BezierCurve = 0;

	TArray<USceneComponent*>	m_ControlPoints;


};
