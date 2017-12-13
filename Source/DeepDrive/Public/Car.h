// 

#pragma once

#include "GameFramework/WheeledVehicle.h"
#include "SplineTrajectory.h"
#include "DataTransferStructs.h"
#include "Car.generated.h"

/**
 * 
 */
UCLASS()
class DEEPDRIVE_API ACar : public AWheeledVehicle
{
	GENERATED_BODY()
	
public:

	ACar();

	virtual void BeginPlay() override;

	virtual void Tick(float DeltaTime) override;

//// "Car controls"

	// steering applied by SetSteering function last time
	float LastAppliedSteering;

	// throttle applied by SetThrottle function last time
	float LastAppliedThrottle;

	// brake applied by SetBrake function last time
	float LastAppliedBrake;

	// handbrake applied by SetHandbrake function last time
	UPROPERTY(BlueprintReadOnly, Category = "Car controls")
	bool LastAppliedHandbrake;

	UFUNCTION(BlueprintCallable, Category = "Car controls")
	void SetSteering(float NewSteering);

	UFUNCTION(BlueprintCallable, Category = "Car controls")
	float GetPhysicsSteering() const;

	UFUNCTION(BlueprintCallable, Category = "Car controls")
	float GetInputSteering() const;

	UFUNCTION(BlueprintCallable, Category = "Car controls")
	void SetThrottle(float NewThrottle);

	UFUNCTION(BlueprintCallable, Category = "Car controls")
	float GetPhysicsThrottle() const;

	UFUNCTION(BlueprintCallable, Category = "Car controls")
	float GetInputThrottle() const;

	UFUNCTION(BlueprintCallable, Category = "Car controls")
	void SetBrake(float NewBrake);

	UFUNCTION(BlueprintCallable, Category = "Car controls")
	float GetPhysicsBrake() const;

	UFUNCTION(BlueprintCallable, Category = "Car controls")
	float GetInputBrake() const;

	UFUNCTION(BlueprintCallable, Category = "Car controls")
	void SetHandbrake(bool NewHandbrake);

	UFUNCTION(BlueprintCallable, Category = "Car controls")
	float GetPhysicsHandbrake() const;

	UFUNCTION(BlueprintCallable, Category = "Car controls")
	bool GetInputHandbrake() const;

	UFUNCTION(BlueprintCallable, Category = "Car controls")
	void SetIsGameDriving(bool NewbIsGameDriving);

	UFUNCTION(BlueprintCallable, Category = "Car controls")
	bool GetIsGameDriving() const;

	UFUNCTION(BlueprintCallable, Category = "Car controls")
	void SetControls(float Steering, float Throttle, float Brake, float Handbrake);

//// data transfer

	// FUnrealToPythonDataLoad contains huge image data. structure shall be transferred between functions with caution
	FUnrealToPythonDataLoad GenerateOutputData() const;

	FVector PreviousTickVelocity; // cm / s

	FVector PreviousTickAngularVelocity; // deg / s

	UPROPERTY(BlueprintReadOnly, Category = "Car physics data")
	FVector Dimension;

	UPROPERTY(BlueprintReadOnly, Category = "Car physics data")
	FVector Acceleration; // cm / s

	UPROPERTY(BlueprintReadOnly, Category = "Car physics data")
	FVector AngularAcceleration; // cm / s

	FVector GetAcceleration() const;

	UFUNCTION(BlueprintCallable, Category = "Car physics data")
	FVector GetAngularVelocity() const;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Car physics data")
	bool bShouldReset;

	bool bPreviousShouldReset;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Car physics data")
	bool bShouldResetPosition;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Car physics data")
	bool bIsResetting;

	UFUNCTION(BlueprintCallable, Category = "Car physics data")
	void SetShouldReset(bool NewbShouldReset);

	UFUNCTION(BlueprintCallable, Category = "Car physics data")
	bool GetShouldReset() const;

	UFUNCTION(BlueprintImplementableEvent, Category = "Car physics data")
	void ResetAgentFinished();

	UFUNCTION(BlueprintCallable, Category = "Car physics data")
	void SetIsResetting(bool NewbIsResetting);

	UFUNCTION(BlueprintCallable, Category = "Car physics data")
	bool GetIsResetting() const;

	UPROPERTY(BlueprintReadOnly, Category = "Reward data")
	float DistanceAlongRoute; // cm

	UPROPERTY(BlueprintReadOnly, Category = "Reward data")
	float DistanceToCenterOfLane; // cm

	UPROPERTY(BlueprintReadOnly, Category = "Reward data")
	int LapNumber;

	FVector GetAngularAcceleration() const;

	double EpisodeStartTime; // microseconds
	
	double EpisodeEndTime; // microseconds

	double GetAgentTime() const;

	double GetEpisodeStartTime() const;
	
	double GetEpisodeEndTime() const;


//// AI

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AI")
	bool bIsGameDriving;

	TWeakObjectPtr<ASplineTrajectory> Trajectory;

	UPROPERTY(VisibleAnywhere, Category = "AI")
	float WaypointDistanceAlongSpline;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AI")
	float WaypointStep;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AI")
	float CloseDistanceThreshold;  // TODO: Figure out where this is changed (in blueprints?) to 1500 and cleanup the start progress to be closer to zero

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AI")
	float AIThrottle;

	void MoveAlongSpline();

	bool UpdateSplineProgress();

	bool searchAlongSpline(FVector CurrentLocation, int step, float distToCurrent, float& distanceAlongRoute);

	bool getDistanceAlongSplineAtLocationWithStep(FVector CurrentLocation, unsigned int step, float& distanceAlongRoute);

	void GetDistanceAlongRouteAtLocation(FVector CurrentLocation);

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AI")
	bool IsInputEnabled = false;

};
