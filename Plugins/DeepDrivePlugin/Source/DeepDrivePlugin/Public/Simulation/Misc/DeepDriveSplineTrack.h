

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "DeepDriveSplineTrack.generated.h"

class USplineComponent;
class ADeepDriveAgent;


/**
 * 
 */

UCLASS()
class DEEPDRIVEPLUGIN_API ADeepDriveSplineTrack	:	public AActor
{
	GENERATED_BODY()

	//	maps agent to current key on spline
	typedef TMap<ADeepDriveAgent*, float>		TAgentMap;

	struct AgentData
	{
		AgentData(ADeepDriveAgent *a, float k, float d)
			:	agent(a)
			,	key(k)
			,	distance(d)
		{}

		ADeepDriveAgent		*agent;
		float				key;
		float				distance;
	};


public:

	ADeepDriveSplineTrack();

	~ADeepDriveSplineTrack();

	virtual void BeginPlay() override;

	virtual void Tick(float DeltaTime) override;


	void setBaseLocation(const FVector &baseLocation);

	FVector getLocationAhead(float distanceAhead, float sideOffset);

	float getSpeedLimit(float distanceAhead);

	void registerAgent(ADeepDriveAgent &agent, float curKey);

	bool getNextAgent(ADeepDriveAgent &agent, ADeepDriveAgent* &agentPtr, float &distance);

	void getPreviousAgent(const FVector &location, ADeepDriveAgent* &agentPtr, float &distance);

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Track")
	TArray<FVector2D>		SpeedLimits;

	UFUNCTION(BlueprintCallable, Category = "Track")
	USplineComponent* GetSpline();

protected:

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Track")
	USplineComponent		*SplineTrack = 0;

private:

	float getInputKeyAhead(float distanceAhead);

	float getDistance(float key);

	TArray<FVector2D>				m_SpeedLimits;

	FVector							m_BaseLocation;
	float							m_BaseKey = 0.0f;

	TArray<AgentData>				m_RegisteredAgents;
	
	float							m_TrackLength;
};

inline USplineComponent* ADeepDriveSplineTrack::GetSpline()
{
	return SplineTrack;
}
