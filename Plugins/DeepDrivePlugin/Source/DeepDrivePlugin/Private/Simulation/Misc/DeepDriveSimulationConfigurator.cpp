

#include "DeepDrivePlugin.h"
#include "DeepDriveSimulationConfigurator.h"

#include "Public/Simulation/DeepDriveSimulationDefines.h"
#include "Public/Simulation/DeepDriveSimulation.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoadNetworkComponent.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoadLinkProxy.h"
#include "Public/Simulation/RoadNetwork/DeepDriveRoadSegmentProxy.h"

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

			if(ScenarioIndex >= 0 && ScenarioIndex < Scenarios.Num() && Agents.Num() > 0)
			{
				const FDeepDriveSimulationScenario &scenario = Scenarios[ScenarioIndex];

				FDeepDriveScenarioConfiguration scenarioConfig;
				scenarioConfig.EgoAgent.Agent = Agents[FMath::RandRange(0, Agents.Num() - 1)];
				scenarioConfig.EgoAgent.StartPosition = resolvePosition(scenario.EgoAgent.Start);
				scenarioConfig.EgoAgent.EndPosition = resolvePosition(scenario.EgoAgent.Destination);
				scenarioConfig.EgoAgent.MaxSpeed = FMath::RandRange(scenario.EgoAgent.MinSpeed, scenario.EgoAgent.MaxSpeed);

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
