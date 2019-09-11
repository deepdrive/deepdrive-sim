
#include "DeepDrivePlugin.h"
#include "Simulation/TrafficLight/DeepDriveTrafficLightsCtrlCmp.h"

#include "Simulation/TrafficLight/DeepDriveTrafficLight.h"

// Sets default values for this component's properties
UDeepDriveTrafficLightsCtrlCmp::UDeepDriveTrafficLightsCtrlCmp()
{
	// Set this component to be initialized when the game starts, and to be ticked every frame.  You can turn these features
	// off to improve performance if you don't need them.
	PrimaryComponentTick.bCanEverTick = true;

	// ...
}


// Called when the game starts
void UDeepDriveTrafficLightsCtrlCmp::BeginPlay()
{
	Super::BeginPlay();
	
	float totalDuration = 0.0f;
	for(auto &cycle : Cycles)
		totalDuration += cycle.Duration;

	m_curCycleTime = FMath::Fmod(PhaseLag, totalDuration);
	m_curCycleIndex = 0;
	uint32 curIndex = 0;
	for(auto &cycle : Cycles)
	{
		if	(	m_curCycleTime >= 0.0f
			&&	m_curCycleTime < cycle.Duration
			)
		{
			for(auto &circuit : cycle.Circuits)
			{
				if(m_curCycleTime < circuit.Delay)
				{
					circuit.setState(-1);
					for(auto &trafficLight : circuit.TrafficLights)
						trafficLight->SetToRed(-1.0f);
				}
				else if(m_curCycleTime < (circuit.Delay + circuit.Duration) )
				{
					circuit.setState(0);
					for(auto &trafficLight : circuit.TrafficLights)
						trafficLight->SetToGreen(m_curCycleTime - circuit.Delay);
				}
				else
				{
					circuit.setState(1);
					for(auto &trafficLight : circuit.TrafficLights)
						trafficLight->SetToRed(m_curCycleTime - circuit.Delay - circuit.Duration);
				}
			}
			m_curCycleIndex = curIndex;
		}

		for(auto &circuit : cycle.Circuits)
		{
			circuit.setState(-1);
			for(auto &trafficLight : circuit.TrafficLights)
			{
				trafficLight->SetToRed(-1.0f);
			}
		}

		m_curCycleTime -= cycle.Duration;
	}
}


// Called every frame
void UDeepDriveTrafficLightsCtrlCmp::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	if(m_curCycleIndex < Cycles.Num())
	{
		m_curCycleTime += DeltaTime;
		if(m_curCycleTime > Cycles[m_curCycleIndex].Duration)
		{
			m_curCycleTime = m_curCycleTime - Cycles[m_curCycleIndex].Duration;
			m_curCycleIndex = (m_curCycleIndex + 1) % Cycles.Num();
		}

		FDeepDriveTrafficLightCycle &cycle = Cycles[m_curCycleIndex];
		for(auto &circuit : cycle.Circuits)
		{
			const int8 curState = circuit.getState();
			if(curState < 0)
			{
				if	(	m_curCycleTime > circuit.Delay
					&&	m_curCycleTime < (circuit.Delay + circuit.Duration)
					)
				{
					circuit.setState(0);
					for(auto &trafficLight : circuit.TrafficLights)
						trafficLight->SwitchToGreen();
				}
			}
			else if(curState == 0)
			{
				if(m_curCycleTime > (circuit.Delay + circuit.Duration))
				{
					circuit.setState(1);
					for (auto &trafficLight : circuit.TrafficLights)
						trafficLight->SwitchToRed();
				}
			}
		}
	}
}
