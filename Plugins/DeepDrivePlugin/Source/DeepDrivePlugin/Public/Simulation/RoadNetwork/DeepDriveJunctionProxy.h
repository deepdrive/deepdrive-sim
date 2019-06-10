

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoadNetwork.h"
#include "DeepDriveJunctionProxy.generated.h"

class ADeepDriveRoadSegmentProxy;
class ADeepDriveRoadLinkProxy;
class UBezierCurveComponent;

USTRUCT(BlueprintType)
struct FDeepDriveLaneConnectionCustomCurveParams
{
	GENERATED_USTRUCT_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	float		Parameter0 = 0.0f;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	float		Parameter1 = 0.0f;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	float		Parameter2 = 0.0f;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	float		Parameter3 = 0.0f;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	float		Parameter4 = 0.0f;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	float		Parameter5 = 0.0f;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	float		Parameter6 = 0.0f;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	float		Parameter7 = 0.0f;

};

USTRUCT(BlueprintType)
struct FDeepDriveLaneConnectionProxy
{
	GENERATED_USTRUCT_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	ADeepDriveRoadLinkProxy *FromLink = 0;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	ADeepDriveRoadSegmentProxy	*FromSegment = 0;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	ADeepDriveRoadLinkProxy *ToLink = 0;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	ADeepDriveRoadSegmentProxy	*ToSegment = 0;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	EDeepDriveConnectionShape	ConnectionShape = EDeepDriveConnectionShape::STRAIGHT_LINE;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	float	SpeedLimit = 25.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	float	SlowDownDistance = 1000.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	FDeepDriveLaneConnectionCustomCurveParams	CustomCurveParams;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	ADeepDriveRoadSegmentProxy	*ConnectionSegment = 0;

};

USTRUCT(BlueprintType)
struct FDeepDriveTurningRestriction
{
	GENERATED_USTRUCT_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	ADeepDriveRoadLinkProxy *FromLink = 0;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	ADeepDriveRoadLinkProxy *ToLink = 0;

};

UCLASS()
class DEEPDRIVEPLUGIN_API ADeepDriveJunctionProxy : public AActor
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	ADeepDriveJunctionProxy();

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

	virtual bool ShouldTickIfViewportsOnly() const override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

	const TArray<ADeepDriveRoadLinkProxy*>& getLinksIn();

	const TArray<ADeepDriveRoadLinkProxy*>& getLinksOut();

	const TArray<FDeepDriveLaneConnectionProxy>& getLaneConnections();

	const TArray<FDeepDriveTurningRestriction>& getTurningRestrictions();

protected:

	UPROPERTY(EditDefaultsOnly, Category = Default)
	USceneComponent				*Root = 0;

	UPROPERTY(EditAnywhere, Category = Configuration)
	TArray<ADeepDriveRoadLinkProxy*>	LinksIn;

	UPROPERTY(EditAnywhere, Category = Configuration)
	TArray<ADeepDriveRoadLinkProxy*>	LinksOut;

	UPROPERTY(EditAnywhere, Category = Configuration)
	TArray<FDeepDriveLaneConnectionProxy>	LaneConnections;

	UPROPERTY(EditAnywhere, Category = Configuration)
	TArray<FDeepDriveTurningRestriction>	TurningRestrictions;

	UPROPERTY(EditAnywhere, Category = Debug)
	FColor	JunctionColor = FColor(0, 255, 0, 64);

	UPROPERTY(EditAnywhere, Category = Debug)
	FColor	ConnectionColor = FColor(255, 0, 255, 255);

private:

	bool extractConnection(const FDeepDriveLaneConnectionProxy &connectionProxy, FVector &fromStart, FVector &fromEnd, FVector &toStart, FVector &toEnd);
	void drawQuadraticConnectionSegment(const FVector &fromStart, const FVector &fromEnd, const FVector &toStart, const FVector &toEnd, const FDeepDriveLaneConnectionCustomCurveParams &params);
	void drawCubicConnectionSegment(const FVector &fromStart, const FVector &fromEnd, const FVector &toStart, const FVector &toEnd, const FDeepDriveLaneConnectionCustomCurveParams &params);
	void drawUTurnConnectionSegment(const FVector &fromStart, const FVector &fromEnd, const FVector &toStart, const FVector &toEnd, const FDeepDriveLaneConnectionCustomCurveParams &params);

	void drawAutoTurnRestriction(uint8 prio);

	UPROPERTY()
	UBezierCurveComponent		*m_BezierCurve = 0;

	bool						m_IsGameRunning = false;

	const uint8					m_DrawPrioConnection = 120;
};

inline const TArray<ADeepDriveRoadLinkProxy*>& ADeepDriveJunctionProxy::getLinksIn()
{
	return LinksIn;
}

inline const TArray<ADeepDriveRoadLinkProxy*>& ADeepDriveJunctionProxy::getLinksOut()
{
	return LinksOut;
}

inline const TArray<FDeepDriveLaneConnectionProxy>& ADeepDriveJunctionProxy::getLaneConnections()
{
	return LaneConnections;
}

inline const TArray<FDeepDriveTurningRestriction>& ADeepDriveJunctionProxy::getTurningRestrictions()
{
	return TurningRestrictions;
}
