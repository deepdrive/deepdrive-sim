

#include "DeepDrivePlugin.h"
#include "DeepDriveSimulationConfigurator.h"

#include "Public/Simulation/DeepDriveSimulationDefines.h"
#include "Public/Simulation/DeepDriveSimulation.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoadNetworkComponent.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoadLinkProxy.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoadSegmentProxy.h"

#include "Runtime/Engine/Classes/Commandlets/Commandlet.h"

// Sets default values
ADeepDriveSimulationConfigurator::ADeepDriveSimulationConfigurator()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

}

// Called when the game starts or when spawned
void ADeepDriveSimulationConfigurator::BeginPlay()
{
	Super::BeginPlay();

	m_StartCounter = 2;	

	{
#if WITH_EDITOR
		m_ScenarioIndex = InitialScenarioIndex;
		m_isRemotelyControlled = InitialRemotelyControlled;
#else
		m_ScenarioIndex = -1;
#endif
		TArray<FString> tokens;
		TArray<FString> switches;
		TMap<FString, FString> params;
		UCommandlet::ParseCommandLine(FCommandLine::Get(), tokens, switches, params);
		UE_LOG(LogDeepDriveSimulation, Log, TEXT("Scenario:  %d %d %d"), tokens.Num(), switches.Num(), params.Num() );

		if(params.Contains("scenario_index"))
		{
			m_ScenarioIndex = FCString::Atoi(*(params["scenario_index"]));
			UE_LOG(LogDeepDriveSimulation, Log, TEXT("Found it %d"), m_ScenarioIndex);
		}
		for (auto &s : switches)
		{
			if (s == "remote_ai")
			{
				m_isRemotelyControlled = true;
				break;
			}
		}
	}
}

// Called every frame
void ADeepDriveSimulationConfigurator::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	if(Simulation && Simulation->ScenarioMode)
	{
		if(m_StartCounter > 0)
			--m_StartCounter;

		if(m_StartCounter == 0)
		{
			m_StartCounter = -1;

			if(m_ScenarioIndex >= 0 && m_ScenarioIndex < Scenarios.Num() && Agents.Num() > 0)
			{
				const FDeepDriveSimulationScenario &scenario = Scenarios[m_ScenarioIndex];

				FDeepDriveScenarioConfiguration scenarioConfig;
				scenarioConfig.EgoAgent.Agent = Agents[FMath::RandRange(0, Agents.Num() - 1)];
				scenarioConfig.EgoAgent.StartPosition = resolvePosition(scenario.EgoAgent.Start);
				scenarioConfig.EgoAgent.EndPosition = resolvePosition(scenario.EgoAgent.Destination);
				scenarioConfig.EgoAgent.MaxSpeed = FMath::RandRange(scenario.EgoAgent.MinSpeed, scenario.EgoAgent.MaxSpeed);
				scenarioConfig.EgoAgent.IsRemotelyControlled = m_isRemotelyControlled;

				for(auto &scenarioAgent : scenario.AdditionalAgents)
				{
					scenarioConfig.Agents.Add(FDeepDriveAgentScenarioConfiguration());
					FDeepDriveAgentScenarioConfiguration &dstCfg = scenarioConfig.Agents.Last();
					dstCfg.Agent = Agents[FMath::RandRange(0, Agents.Num() - 1)];
					dstCfg.StartPosition = resolvePosition(scenarioAgent.Start);
					dstCfg.EndPosition = resolvePosition(scenarioAgent.Destination);
					dstCfg.MaxSpeed = FMath::RandRange(scenarioAgent.MinSpeed, scenarioAgent.MaxSpeed);
				}

				Simulation->ConfigureSimulation(scenarioConfig);
			}

		}
	}


}


FVector ADeepDriveSimulationConfigurator::resolvePosition(const FDeepDriveRoadLocation &roadLocation)
{
	FVector position = FVector::ZeroVector;
	if(roadLocation.Segment)
	{
		const USplineComponent *spline = roadLocation.Segment->getSpline();
		if(spline && spline->GetNumberOfSplinePoints() > 2)
		{
			const float relDist = spline->GetSplineLength() * roadLocation.RelativePosition;
			position = spline->GetLocationAtDistanceAlongSpline(relDist, ESplineCoordinateSpace::World);
		}
		else
		{
			position = roadLocation.Segment->getStartPoint();
			FVector dir = roadLocation.Segment->getEndPoint() - position;
			const float length = dir.Size();
			dir.Normalize();
			dir *= roadLocation.RelativePosition * length;
			position += dir;
		}
	}
	else if(roadLocation.Link)
	{
		position = roadLocation.Link->getStartPoint();
		FVector dir = roadLocation.Link->getEndPoint() - position;
		const float length = dir.Size();
		dir.Normalize();
		dir *= roadLocation.RelativePosition * length;
		position += dir;
	}

	return position;
}
