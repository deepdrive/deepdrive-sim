

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "DeepDriveSplineTrack.generated.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveSplineTrack, Log, All);

class USplineComponent;
class ADeepDriveAgent;
class UDeepDriveRandomStream;

/**
 * 
 */

UCLASS(Blueprintable)
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

	virtual void PostInitializeComponents() override;

	virtual void Tick(float DeltaTime) override;

	void setBaseLocation(const FVector &baseLocation);

	FVector getLocationAhead(float distanceAhead, float sideOffset);

	float getSpeedLimit(float distanceAhead);

	void registerAgent(ADeepDriveAgent &agent, float curKey);

	void unregisterAgent(ADeepDriveAgent &agent);

	bool getNextAgent(ADeepDriveAgent &agent, ADeepDriveAgent* &agentPtr, float &distance);

	void getPreviousAgent(const FVector &location, ADeepDriveAgent* &agentPtr, float &distance);

	UFUNCTION(BlueprintCallable, Category = "Track")
	void AddSpeedLimit(float Distance, float SpeedLimit);

	UFUNCTION(BlueprintCallable, Category = "Track")
	USplineComponent* GetSpline();

	UPROPERTY(EditAnywhere, Category = "Track")
	ADeepDriveSplineTrack	*OppositeTrack = 0;

	UPROPERTY(EditAnywhere, Category = "Track")
	float	RandomSlotDistance = 2000.0f;

	float getRandomDistanceAlongTrack(FRandomStream &randomStream);

	float getRandomDistanceAlongTrack(UDeepDriveRandomStream &randomStream);

protected:

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Track")
	USplineComponent		*SplineTrack = 0;

private:

	void exportAsTextFile(const FString &fileName, float steppingDistance, float sideOffset);

	void exportControlPoints(const FString &fileName);

	void importControlPoints(const FString &fileName);

	float getInputKeyAhead(float distanceAhead);

	float getDistance(float key);

	TArray<FVector2D>				m_SpeedLimits;
	bool							m_SpeedLimitsDirty = false;

	FVector							m_BaseLocation;
	float							m_BaseKey = 0.0f;

	TArray<AgentData>				m_RegisteredAgents;

	float							m_TrackLength;

	int32							m_RandomSlotCount = 0;
	int32							m_remainingSlots = 0;
	TSet<int32>						m_RandomSlots;
};

inline USplineComponent* ADeepDriveSplineTrack::GetSpline()
{
	return SplineTrack;
}
