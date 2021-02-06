

#include "DeepDriveSplineTrack.h"
#include "Runtime/Engine/Classes/Components/SplineComponent.h"
#include "Simulation/Agent/DeepDriveAgent.h"
#include "Simulation/Misc/DeepDriveRandomStream.h"

#include <fstream>

DEFINE_LOG_CATEGORY(LogDeepDriveSplineTrack);

static FString getCtrlPointsName(const FString &track)
{
	FString basePath( FPaths::ProjectContentDir() + "DeepDrive/Maps/Sublevels/");
	return basePath + "DeepDrive_Canyon_" + track + FString(".ctrl");
}

ADeepDriveSplineTrack::ADeepDriveSplineTrack()
	:	SplineTrack(0)
{
	SplineTrack = CreateDefaultSubobject<USplineComponent>(TEXT("SplineTrack"));
	RootComponent = SplineTrack;

	PrimaryActorTick.bCanEverTick = true;
	PrimaryActorTick.bStartWithTickEnabled = true;
	PrimaryActorTick.TickGroup = TG_PrePhysics;

	importControlPoints(getCtrlPointsName(GetName()));
	UE_LOG(LogDeepDriveSplineTrack, Log, TEXT("ADeepDriveSplineTrack::ADeepDriveSplineTrack %d"), SplineTrack->GetNumberOfSplinePoints() );
}

ADeepDriveSplineTrack::~ADeepDriveSplineTrack()
{
}

void ADeepDriveSplineTrack::OnConstruction(const FTransform & Transform)
{
	importControlPoints(getCtrlPointsName(GetName()) );
	UE_LOG(LogDeepDriveSplineTrack, Log, TEXT("ADeepDriveSplineTrack::OnConstruction") );
}

void ADeepDriveSplineTrack::PostInitializeComponents()
{
	Super::PostInitializeComponents();

	UE_LOG(LogDeepDriveSplineTrack, Log, TEXT("ADeepDriveSplineTrack::PostInitializeComponents %d"), SplineTrack->GetNumberOfSplinePoints() );

	SetActorTickEnabled(true);

	PrimaryActorTick.TickGroup = TG_PrePhysics;
	m_TrackLength = SplineTrack->GetSplineLength();

	m_RandomSlotCount = FGenericPlatformMath::FloorToInt(m_TrackLength / RandomSlotDistance);
	m_remainingSlots = m_RandomSlotCount;

	UE_LOG(LogDeepDriveSplineTrack, Log, TEXT("Track %s Length %f Slot Count %d"), *(GetName()), m_TrackLength, m_RandomSlotCount );

	// exportControlPoints( ctrlPointsName );
	importControlPoints(getCtrlPointsName(GetName()) );
}

void ADeepDriveSplineTrack::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	for (auto &item : m_RegisteredAgents)
	{
		item.key = SplineTrack->FindInputKeyClosestToWorldLocation(item.agent->GetActorLocation());
		item.distance = getDistance(item.key);

		//UE_LOG(LogDeepDriveSplineTrack, Log, TEXT("Agent %d Distance %f"), item.agent->getAgentId(), item.distance );
	}

	if (m_RegisteredAgents.Num() == 1)
	{
		m_RegisteredAgents[0].agent->setNextAgent(0, -1.0f);
		m_RegisteredAgents[0].agent->setPrevAgent(0, -1.0f);
	}
	else if (m_RegisteredAgents.Num() > 1)
	{
		m_RegisteredAgents.Sort([](const AgentData &lhs, const AgentData &rhs) {	return lhs.key < rhs.key;	 });

		if (m_RegisteredAgents.Num() == 2)
		{
			ADeepDriveAgent *agent0 = m_RegisteredAgents[0].agent;
			ADeepDriveAgent *agent1 = m_RegisteredAgents[1].agent;
			float dist01 = FMath::Max(0.0f, m_RegisteredAgents[1].distance - m_RegisteredAgents[0].distance - agent0->getFrontBumperDistance() - agent1->getBackBumperDistance());
			float dist10 = FMath::Max(0.0f, (m_TrackLength - m_RegisteredAgents[1].distance + m_RegisteredAgents[0].distance) - agent1->getFrontBumperDistance() - agent0->getBackBumperDistance());

			agent0->setNextAgent(agent1, dist01);
			agent0->setPrevAgent(agent1, dist10);

			agent1->setNextAgent(agent0, dist10);
			agent1->setPrevAgent(agent0, dist01);
		}
		else
		{
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

void ADeepDriveSplineTrack::AddSpeedLimit(float Distance, float SpeedLimit)
{
	const FVector loc = SplineTrack->GetLocationAtDistanceAlongSpline(Distance, ESplineCoordinateSpace::World);
	m_SpeedLimits.Add(FVector2D(SplineTrack->FindInputKeyClosestToWorldLocation(loc), SpeedLimit));
	m_SpeedLimitsDirty = true;
}

float ADeepDriveSplineTrack::getSpeedLimit(float distanceAhead)
{
	float speedLimit = -1.0f;

	if(m_SpeedLimitsDirty)
	{
		m_SpeedLimits.Sort([](const FVector2D &lhs, const FVector2D &rhs) { return lhs.X < rhs.X; });
		m_SpeedLimitsDirty = false;
	}

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

void ADeepDriveSplineTrack::unregisterAgent(ADeepDriveAgent &agent)
{
	int32 i = 0;
	for(; i < m_RegisteredAgents.Num(); ++i)
		if(m_RegisteredAgents[i].agent == &agent)
		break;

	if(i < m_RegisteredAgents.Num())
		m_RegisteredAgents.RemoveAt(i);
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
	agentPtr = 0;
	distance = -1.0f;

	float key = SplineTrack->FindInputKeyClosestToWorldLocation(location);

	if(m_RegisteredAgents.Num() == 1)
	{
		agentPtr = m_RegisteredAgents[0].agent;
		const float dist = m_RegisteredAgents[0].distance;
		const float curDistance = getDistance(key);
		if(key > m_RegisteredAgents[0].key)
		{
			distance = curDistance - dist;
		}
		else
		{
			distance = curDistance - (m_TrackLength - dist);
		}
		//UE_LOG(LogDeepDriveSplineTrack, Log, TEXT("Agent %d Distance %f"), agentPtr->getAgentId(), distance );
	}
	else if(m_RegisteredAgents.Num() > 1)
	{
		int32 ind = m_RegisteredAgents.Num() - 1;
		for (int32 i = 0; i < m_RegisteredAgents.Num(); ++i)
		{
			if (m_RegisteredAgents[i].key < key)
			{
				ind = i;
			}
			else
				break;
		}

		agentPtr = m_RegisteredAgents[ind].agent;
		const float dist = m_RegisteredAgents[ind].distance;
		const float curDistance = getDistance(key);
		if(curDistance >= dist)
		{
			distance = curDistance - dist;
		}
		else
		{
			distance = m_TrackLength - dist + curDistance;
		}

	}
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

void ADeepDriveSplineTrack::resetRandomSlots()
{
	m_remainingSlots = m_RandomSlotCount;
	m_RandomSlots.Empty();
}

float ADeepDriveSplineTrack::getRandomDistanceAlongTrack(FRandomStream &randomStream)
{
	float distance = -1.0f;

	if(m_remainingSlots > 0)
	{
		int32 randomSlot = 0;

		do
		{
			randomSlot = randomStream.RandRange(0, m_RandomSlotCount);
		} while(m_RandomSlots.Contains(randomSlot) == true);

		m_RandomSlots.Add(randomSlot);
		--m_remainingSlots;

		distance = static_cast<float> (randomSlot) * RandomSlotDistance;

		UE_LOG(LogDeepDriveSplineTrack, Log, TEXT("Random distance %f on %s remaining %d"), distance, *(GetName()), m_remainingSlots);
	}

	SplineTrack->GetLocationAtDistanceAlongSpline( 200.0f, ESplineCoordinateSpace::World);

	return distance;
}


float ADeepDriveSplineTrack::getRandomDistanceAlongTrack(UDeepDriveRandomStream &randomStream)
{
	float distance = -1.0f;

	if (m_remainingSlots > 0)
	{
		int32 randomSlot = 0;

		do
		{
			randomSlot = randomStream.RandomInteger(m_RandomSlotCount);
		} while (m_RandomSlots.Contains(randomSlot) == true);

		m_RandomSlots.Add(randomSlot);
		--m_remainingSlots;

		distance = static_cast<float> (randomSlot) * RandomSlotDistance;

		UE_LOG(LogDeepDriveSplineTrack, Log, TEXT("Random distance %f on %s remaining %d"), distance, *(GetName()), m_remainingSlots);
	}

	SplineTrack->GetLocationAtDistanceAlongSpline(200.0f, ESplineCoordinateSpace::World);

	return distance;
}

void ADeepDriveSplineTrack::exportAsTextFile(const FString &fileName, float steppingDistance, float sideOffset)
{
	if(SplineTrack && SplineTrack->GetNumberOfSplinePoints() > 0)
	{
		std::ofstream outStream( TCHAR_TO_ANSI(*fileName) );
		if (outStream.good())
		{
			const float trackLength = SplineTrack->GetSplineLength();

			outStream << "#	X,Y,Z\n";
			for(float curDistance = 0.0f; curDistance < trackLength; curDistance += steppingDistance)
			{
				FVector loc = SplineTrack->GetLocationAtDistanceAlongSpline(curDistance, ESplineCoordinateSpace::World);
				if(sideOffset != 0.0f)
				{
					FVector right = SplineTrack->GetRightVectorAtDistanceAlongSpline(curDistance, ESplineCoordinateSpace::World);
					loc += right * sideOffset;
				}

				outStream	<<	loc.X << "," <<	loc.Y << "," <<	loc.Z
							<< "\n";

				// outStream	<<	"(X=" << loc.X << ",Y=" <<	loc.Y << ",Z=" <<	loc.Z
				// 			<< "),\n";

			}
			UE_LOG(LogDeepDriveSplineTrack, Log, TEXT("Exported track to %s"), *(fileName));
		}
	}

}

void ADeepDriveSplineTrack::exportTrack()
{
	if(CenterSplineFile.Len())
	{
		exportAsTextFile(getFullExportName(CenterSplineFile) , 100.0f, CenterSplineOffset );
	}
	if(LeftSplineFile.Len())
	{
		exportAsTextFile(getFullExportName(LeftSplineFile) , 100.0f, -FGenericPlatformMath::Abs(LeftSplineOffset) );
	}
	if(RightSplineFile.Len())
	{
		exportAsTextFile(getFullExportName(RightSplineFile) , 100.0f, FGenericPlatformMath::Abs(RightSplineOffset) );
	}
}

void ADeepDriveSplineTrack::exportControlPoints(const FString &fileName)
{
	if(SplineTrack && SplineTrack->GetNumberOfSplinePoints() > 0)
	{
		std::ofstream outStream( TCHAR_TO_ANSI(*fileName) );
		if (outStream.good())
		{
			// outStream << "#	Location(X,Y,Z), Rotation(Pitch,Roll,Yaw) ArriveTangent(X,Y,Z) LeaveTangent(X,Y,Z)\n";

			for(signed i = 0; i < SplineTrack->GetNumberOfSplinePoints(); ++i)
			{
				ESplinePointType::Type type = SplineTrack->GetSplinePointType(i);
				FVector loc = SplineTrack->GetLocationAtSplinePoint(i, ESplineCoordinateSpace::Local);
				FRotator rot = SplineTrack->GetRotationAtSplinePoint(i, ESplineCoordinateSpace::Local);
				FVector aTng = SplineTrack->GetArriveTangentAtSplinePoint(i, ESplineCoordinateSpace::Local);
				FVector lTng = SplineTrack->GetLeaveTangentAtSplinePoint(i, ESplineCoordinateSpace::Local);

				// outStream	<<	loc.X << "," <<	loc.Y << "," <<	loc.Z
				// 			<<	" " << rot.Pitch << "," <<	rot.Roll << "," <<	rot.Yaw
				// 			<<	" " << aTng.X << "," <<	aTng.Y << "," << lTng.Z
				// 			<<	" " << lTng.X << "," <<	lTng.Y << "," << lTng.Z
				// 			<< "\n";

				outStream	<< type
							<<	" " << loc.X << " " <<	loc.Y << " " <<	loc.Z
							<<	" " << rot.Pitch << " " <<	rot.Roll << " " <<	rot.Yaw
							<<	" " << aTng.X << " " <<	aTng.Y << " " << lTng.Z
							<<	" " << lTng.X << " " <<	lTng.Y << " " << lTng.Z
							<< "\n";
			}
			UE_LOG(LogDeepDriveSplineTrack, Log, TEXT("Exported control points to %s"), *(fileName));
		}
	}
}

void ADeepDriveSplineTrack::importControlPoints(const FString &fileName)
{
		std::ifstream inStream( TCHAR_TO_ANSI(*fileName) );
		if (inStream.good())
		{
			TArray<FSplinePoint> points;
			for(signed i = 0; i < SplineTrack->GetNumberOfSplinePoints(); ++i)
			{
				int32 type;
				FVector loc;
				FRotator rot;
				FVector aTng;
				FVector lTng;
				
				inStream	>> type
							>> loc.X >> loc.Y >> loc.Z
							>> rot.Pitch >>	rot.Roll >>	rot.Yaw
							>> aTng.X >> aTng.Y >> aTng.Z
							>> lTng.X >> lTng.Y >> lTng.Z;

				FSplinePoint p	( static_cast<float> (i), loc, aTng, lTng, rot, FVector(1.0f, 1.0f, 1.0f), static_cast<ESplinePointType::Type> (type));

				points.Add(p);

				// UE_LOG(LogDeepDriveSplineTrack, Log, TEXT("l (%s) r(%s) a(%s) t(%s)"), *(loc.ToString()), *(rot.ToString()), *(aTng.ToString()), *(lTng.ToString()) );
			}

			if(points.Num() > 2)
			{
				SplineTrack->ClearSplinePoints();
				SplineTrack->AddPoints(points, true);
				UE_LOG(LogDeepDriveSplineTrack, Log, TEXT("Imported control points from %s"), *(fileName));
			}
		}
}

FString ADeepDriveSplineTrack::getFullExportName(const FString &name)
{
	return BasePath.Len() > 0 ? FPaths::Combine(BasePath, name) : name;
}
