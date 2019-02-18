

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "BezierCurveComponent.generated.h"


UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent) )
class DEEPDRIVEPLUGIN_API UBezierCurveComponent : public UActorComponent
{
	GENERATED_BODY()

public:	
	// Sets default values for this component's properties
	UBezierCurveComponent();

protected:
	// Called when the game starts
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

	UFUNCTION(BlueprintCallable, Category = "Default")
	void ClearControlPoints();
	
	UFUNCTION(BlueprintCallable, Category = "Default")
	void AddControlPoint(const FVector &ControlPoint);
	
	UFUNCTION(BlueprintCallable, Category = "Default")
	void UpdateControlPoint(int32 Index, const FVector &ControlPoint);

	UFUNCTION(BlueprintCallable, Category = "Default")
	FVector Evaluate(float T);

protected:

	FVector evaluateQuadraticSpline(float t);
	FVector evaluateCubicSpline(float t);

	FVector splitDeCasteljau(const TArray<FVector> &points, float t);

	TArray<FVector>			m_ControlPoints;

};
