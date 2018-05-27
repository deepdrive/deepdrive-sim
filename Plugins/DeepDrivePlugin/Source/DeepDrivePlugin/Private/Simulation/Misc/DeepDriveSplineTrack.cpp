

#include "DeepDrivePluginPrivatePCH.h"
#include "DeepDriveSplineTrack.h"
#include "Runtime/Engine/Classes/Components/SplineComponent.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"

ADeepDriveSplineTrack::ADeepDriveSplineTrack()
	:	SplineTrack(0)
{
	SplineTrack = CreateDefaultSubobject<USplineComponent>(TEXT("SplneTrack"));
	RootComponent = SplineTrack;

	PrimaryActorTick.bCanEverTick = true;
	PrimaryActorTick.bStartWithTickEnabled = true;
	PrimaryActorTick.TickGroup = TG_PrePhysics;
}

ADeepDriveSplineTrack::~ADeepDriveSplineTrack()
{
}

void ADeepDriveSplineTrack::BeginPlay()
{
	Super::BeginPlay();

	for (auto &x : SpeedLimits)
	{
		const float dist = x.X;
		const float key = SplineTrack->GetInputKeyAtDistanceAlongSpline(dist);

		const FVector loc = SplineTrack->GetLocationAtDistanceAlongSpline(dist, ESplineCoordinateSpace::World);

		m_SpeedLimits.Add(FVector2D(SplineTrack->FindInputKeyClosestToWorldLocation(loc), x.Y));
	}

	m_SpeedLimits.Sort([](const FVector2D &lhs, const FVector2D &rhs) { return lhs.X < rhs.X; });


	SetActorTickEnabled(true);
	PrimaryActorTick.TickGroup = TG_PrePhysics;
	m_TrackLength = SplineTrack->GetSplineLength();
}

void ADeepDriveSplineTrack::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	for (auto &item : m_RegisteredAgents)
	{
		item.key = SplineTrack->FindInputKeyClosestToWorldLocation(item.agent->GetActorLocation());
		item.distance = getDistance(item.key);
	}

	if (m_RegisteredAgents.Num() > 1)
	{
		m_RegisteredAgents.Sort([](const AgentData &lhs, const AgentData &rhs) {	return lhs.key < rhs.key;	 });

		int32 ind0 = 0;
		int32 ind1 = 1;
		for (int32 i = 0; i < m_RegisteredAgents.Num(); ++i)
		{
			ADeepDriveAgent *agent0 = m_RegisteredAgents[ind0].agent;
			ADeepDriveAgent *agent1 = m_RegisteredAgents[ind1].agent;
			float distance = ind1 > ind0 ? m_RegisteredAgents[ind1].distance - m_RegisteredAgents[ind0].distance : (m_TrackLength - m_RegisteredAgents[ind1].distance + m_RegisteredAgents[ind0].distance);
			distance -= agent0->getFrontBumperDistance() + agent1->getBackBumperDistance();

			agent0->setNextAgent(agent1, FMath::Max(0.0f, distance));
			agent1->setPrevAgent(agent0, FMath::Max(0.0f, distance));

			ind0 = ind1;
			ind1 = (ind1 + 1) % m_RegisteredAgents.Num();
		}
	}

}


void ADeepDriveSplineTrack::setBaseLocation(const FVector &baseLocation)
{
	m_BaseLocation = baseLocation;
	m_BaseKey = SplineTrack->FindInputKeyClosestToWorldLocation(baseLocation);
}


FVector ADeepDriveSplineTrack::getLocationAhead(float distanceAhead, float sideOffset)
{
	const float curKey = getInputKeyAhead(distanceAhead);
	FVector posAhead = SplineTrack->GetLocationAtSplineInputKey(curKey, ESplineCoordinateSpace::World);

	if (sideOffset != 0.0f)
	{
		FVector dir = SplineTrack->GetDirectionAtSplineInputKey(curKey, ESplineCoordinateSpace::World);
		dir.Z = 0.0f;
		dir.Normalize();
		FVector tng(dir.Y, -dir.X, 0.0f);
		posAhead += tng * sideOffset;
	}

	return posAhead;

}

float ADeepDriveSplineTrack::getSpeedLimit(float distanceAhead)
{
	float speedLimit = -1.0f;

	if (m_SpeedLimits.Num() > 0)
	{
		const float curKey = distanceAhead > 0.0f ? getInputKeyAhead(distanceAhead) : m_BaseKey;
		if	(	curKey < m_SpeedLimits[0].X
			||	curKey >= m_SpeedLimits.Last().X
			)
		{
			speedLimit = m_SpeedLimits.Last().Y;
		}
		else
		{
			for (signed i = 0; i < m_SpeedLimits.Num(); ++i)
			{
				if (curKey >= m_SpeedLimits[i].X)
				{
					speedLimit = m_SpeedLimits[i].Y;
				}
				else
					break;
			}
		}
	}

	return speedLimit;
}

void ADeepDriveSplineTrack::registerAgent(ADeepDriveAgent &agent, float curKey)
{
	m_RegisteredAgents.Add(AgentData(&agent, curKey, getDistance(curKey)));
}

bool ADeepDriveSplineTrack::getNextAgent(ADeepDriveAgent &agent, ADeepDriveAgent* &agentPtr, float &distance)
{
	bool foundIt = false;
	if(m_RegisteredAgents.Num() > 1)
	{
		int32 ind = 0;
		for( ; ind < m_RegisteredAgents.Num() && m_RegisteredAgents[ind].agent != &agent; ++ind) {}
		if (ind < m_RegisteredAgents.Num())
		{
			foundIt = true;

			float dist0 = m_RegisteredAgents[ind].distance;

			ind = (ind + 1) % m_RegisteredAgents.Num();
			agentPtr = m_RegisteredAgents[ind].agent;

			float dist1 = m_RegisteredAgents[ind].distance;

			if (dist1 > dist0)
			{
				distance = dist1 - dist0;
			}
			else
			{
				distance = m_TrackLength - dist0 + dist1;
			}

			distance = FMath::Max(0.0f, distance - agent.getFrontBumperDistance() - agentPtr->getBackBumperDistance());
		}
	}
	return foundIt;
}

void ADeepDriveSplineTrack::getPreviousAgent(const FVector &location, ADeepDriveAgent* &agentPtr, float &distance)
{

}


float ADeepDriveSplineTrack::getInputKeyAhead(float distanceAhead)
{
	float key = m_BaseKey;

	while (true)
	{
		const int32 index0 = floor(key);
		const int32 index1 = ceil(key);

		const float dist0 = SplineTrack->GetDistanceAlongSplineAtSplinePoint(index0);
		const float dist1 = SplineTrack->GetDistanceAlongSplineAtSplinePoint(index1);

		const float dist = (SplineTrack->GetLocationAtSplinePoint(index1, ESplineCoordinateSpace::World) - SplineTrack->GetLocationAtSplinePoint(index0, ESplineCoordinateSpace::World)).Size();

		const float relDistance = distanceAhead / dist;

		const float carryOver = key + relDistance - static_cast<float> (index1);

		if (carryOver > 0.0f)
		{
			distanceAhead -= dist * (static_cast<float> (index1) - key);
			const float newDist = (SplineTrack->GetLocationAtSplinePoint((index1 + 1) % SplineTrack->GetNumberOfSplinePoints(), ESplineCoordinateSpace::World) - SplineTrack->GetLocationAtSplinePoint(index1, ESplineCoordinateSpace::World)).Size();
			const float newRelDist = distanceAhead / newDist;
			key = static_cast<float> (index1) + newRelDist;
			if (newRelDist < 1.0f)
				break;
		}
		else
		{
			key += relDistance;
			break;
		}
	}

	return key;
}

float ADeepDriveSplineTrack::getDistance(float key)
{
	float distance = 0.0f;

	const int32 index0 = floor(key);
	const int32 index1 = floor(key + 1.0f);

	const float dist0 = SplineTrack->GetDistanceAlongSplineAtSplinePoint(index0);
	const float dist1 = SplineTrack->GetDistanceAlongSplineAtSplinePoint(index1);

	return FMath::Lerp(dist0, dist1, key - static_cast<float> (index0));
}

