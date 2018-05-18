

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

	struct AgentQueryData
	{
		ADeepDriveAgent		*agent;
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

	void beginAgentQuery(ADeepDriveAgent &agent, bool oppositeDirection, float maxDistance);

	void beginAgentQuery(const FVector &location, bool oppositeDirection, float maxDistance);

	bool nextAgent(ADeepDriveAgent* &agentPtr, float &distance);

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Track")
	TArray<FVector2D>		SpeedLimits;

	UFUNCTION(BlueprintCallable, Category = "Track")
	USplineComponent* GetSpline();

protected:

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Track")
	USplineComponent		*SplineTrack = 0;

private:

	float getInputKeyAhead(float distanceAhead);

	void buildAgentQueryList(float curKey, bool oppositeDirection, float maxDistance);

	TArray<FVector2D>				m_SpeedLimits;

	FVector							m_BaseLocation;
	float							m_BaseKey = 0.0f;

	TAgentMap						m_RegisteredAgents;
	TArray<AgentQueryData>			m_AgentQueryDataList;

};

inline USplineComponent* ADeepDriveSplineTrack::GetSpline()
{
	return SplineTrack;
}
