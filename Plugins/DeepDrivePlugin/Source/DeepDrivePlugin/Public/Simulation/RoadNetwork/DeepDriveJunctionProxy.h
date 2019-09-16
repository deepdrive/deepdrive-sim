

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoadNetwork.h"
#include "DeepDriveJunctionProxy.generated.h"

class ADeepDriveRoadSegmentProxy;
class ADeepDriveRoadLinkProxy;
class UBezierCurveComponent;
class ADeepDriveTrafficLight;

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
struct FDeepDriveJunctionConnectionProxy
{
	GENERATED_USTRUCT_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	ADeepDriveRoadSegmentProxy	*Segment = 0;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	ADeepDriveRoadLinkProxy *ToLink = 0;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	ADeepDriveRoadSegmentProxy	*ToSegment = 0;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	EDeepDriveConnectionShape	ConnectionShape = EDeepDriveConnectionShape::STRAIGHT_LINE;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	float	SpeedLimit = 25.0f;

	// UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	// float	SlowDownDistance = 1000.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	FDeepDriveLaneConnectionCustomCurveParams	CustomCurveParams;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	ADeepDriveRoadSegmentProxy	*ConnectionSegment = 0;

};

USTRUCT(BlueprintType)
struct FDeepDriveTurnDefinitionProxy
{
	GENERATED_USTRUCT_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	ADeepDriveRoadLinkProxy *ToLink = 0;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	EDeepDriveManeuverType ManeuverType;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	FVector WaitingLocation;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	ADeepDriveTrafficLight *TrafficLight = 0;
};


USTRUCT(BlueprintType)
struct FDeepDriveJunctionEntryProxy
{
	GENERATED_USTRUCT_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	ADeepDriveRoadLinkProxy *Link = 0;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	EDeepDriveJunctionSubType	JunctionSubType = EDeepDriveJunctionSubType::AUTO_DETECT;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	EDeepDriveRightOfWay						RightOfWay = EDeepDriveRightOfWay::YIELD;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	FVector										ManeuverEntryPoint;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	FVector										LineOfSight;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	ADeepDriveTrafficLight 						*TrafficLight = 0;

	UPROPERTY(EditAnywhere, Category = Default)
	TArray<FDeepDriveTurnDefinitionProxy> TurnDefinitions;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	TArray<FDeepDriveJunctionConnectionProxy>	Connections;	

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

	const TArray<ADeepDriveRoadLinkProxy*>& getLinksOut();

	const TArray<FDeepDriveJunctionEntryProxy>& getEntries();

	EDeepDriveJunctionType getJunctionType();

protected:

	UPROPERTY(EditDefaultsOnly, Category = Default)
	USceneComponent				*Root = 0;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Configuration)
	EDeepDriveJunctionType	JunctionType = EDeepDriveJunctionType::FOUR_WAY_JUNCTION;

	UPROPERTY(EditAnywhere, Category = Configuration)
	TArray<ADeepDriveRoadLinkProxy*>	LinksOut;

	UPROPERTY(EditAnywhere, Category = Configuration)
	TArray<FDeepDriveJunctionEntryProxy>	Entries;

	UPROPERTY(EditAnywhere, Category = Debug)
	FColor	JunctionColor = FColor(0, 255, 0, 64);

	UPROPERTY(EditAnywhere, Category = Debug)
	FColor	ConnectionColor = FColor(255, 0, 255, 255);

private:

	bool extractConnection(ADeepDriveRoadLinkProxy *link, const FDeepDriveJunctionConnectionProxy &junctionConnectionProxy, FVector &fromStart, FVector &fromEnd, FVector &toStart, FVector &toEnd);
	void drawQuadraticConnectionSegment(const FVector &fromStart, const FVector &fromEnd, const FVector &toStart, const FVector &toEnd, const FDeepDriveLaneConnectionCustomCurveParams &params);
	void drawCubicConnectionSegment(const FVector &fromStart, const FVector &fromEnd, const FVector &toStart, const FVector &toEnd, const FDeepDriveLaneConnectionCustomCurveParams &params);
	void drawUTurnConnectionSegment(const FVector &fromStart, const FVector &fromEnd, const FVector &toStart, const FVector &toEnd, const FDeepDriveLaneConnectionCustomCurveParams &params);

	void drawAutoTurnRestriction(uint8 prio);

	UPROPERTY()
	UBezierCurveComponent		*m_BezierCurve = 0;

	bool						m_IsGameRunning = false;

	const uint8					m_DrawPrioConnection = 120;
};

inline const TArray<ADeepDriveRoadLinkProxy*>& ADeepDriveJunctionProxy::getLinksOut()
{
	return LinksOut;
}

inline const TArray<FDeepDriveJunctionEntryProxy>& ADeepDriveJunctionProxy::getEntries()
{
	return Entries;
}

inline EDeepDriveJunctionType ADeepDriveJunctionProxy::getJunctionType()
{
	return JunctionType;
}
