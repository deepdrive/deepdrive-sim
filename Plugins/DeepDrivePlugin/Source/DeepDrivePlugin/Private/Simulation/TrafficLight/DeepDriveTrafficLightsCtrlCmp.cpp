
#include "DeepDrivePlugin.h"
#include "Simulation/TrafficLight/DeepDriveTrafficLightsCtrlCmp.h"

#include "Simulation/TrafficLight/DeepDriveTrafficLight.h"

DEFINE_LOG_CATEGORY(LogDeepDriveTrafficLightsCtrlCmp);

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

	float curCycleTime = FMath::Fmod(TimeOffset, totalDuration);
	m_curCycleIndex = 0;
	uint32 curIndex = 0;

	UE_LOG(LogDeepDriveTrafficLightsCtrlCmp, Log, TEXT("BeginPlay: total %f for %d cycles %f"), totalDuration, Cycles.Num(), curCycleTime);

	for(auto &cycle : Cycles)
	{
		if	(	curCycleTime >= 0.0f
			&&	curCycleTime < cycle.Duration
			)
		{
			for(auto &circuit : cycle.Circuits)
			{
				if(curCycleTime < circuit.Delay)
				{
					circuit.resetState();
					UE_LOG(LogDeepDriveTrafficLightsCtrlCmp, Log, TEXT("Before Green: %f"));
					for(auto &trafficLight : circuit.TrafficLights)
						trafficLight->SetToRed(-1.0f);
				}
				else if(curCycleTime < (circuit.Delay + circuit.Duration) )
				{
					const float elapsedTime = curCycleTime - circuit.Delay;
					circuit.setState(0);
					UE_LOG(LogDeepDriveTrafficLightsCtrlCmp, Log, TEXT("Inside Green: %f"), elapsedTime);
					for(auto &trafficLight : circuit.TrafficLights)
						trafficLight->SetToGreen(elapsedTime);
				}
				else
				{
					const float elapsedTime = curCycleTime - circuit.Delay - circuit.Duration;
					circuit.setState(1);
					UE_LOG(LogDeepDriveTrafficLightsCtrlCmp, Log, TEXT("After Green: %f"), elapsedTime);
					for(auto &trafficLight : circuit.TrafficLights)
						trafficLight->SetToRed(elapsedTime);
				}
			}
			m_curCycleIndex = curIndex;
			m_curCycleTime = curCycleTime;
		}
		else
		{
			UE_LOG(LogDeepDriveTrafficLightsCtrlCmp, Log, TEXT("Setting cycle %d to red"), curIndex );
			for (auto &circuit : cycle.Circuits)
			{
				circuit.resetState();
				for (auto &trafficLight : circuit.TrafficLights)
				{
					trafficLight->SetToRed(-1.0f);
					UE_LOG(LogDeepDriveTrafficLightsCtrlCmp, Log, TEXT("  %d"), static_cast<int32> (trafficLight->CurrentPhase) );
				}
			}
		}

		curIndex++;
		curCycleTime -= cycle.Duration;
	}
}


// Called every frame
void UDeepDriveTrafficLightsCtrlCmp::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
#if WITH_EDITOR
	if(TickType == ELevelTick::LEVELTICK_ViewportsOnly)
		return;
#endif

	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	if(m_curCycleIndex < Cycles.Num())
	{
		// UE_LOG(LogDeepDriveTrafficLightsCtrlCmp, Log, TEXT("%f  %d"), m_curCycleTime, m_curCycleIndex);
		m_curCycleTime += DeltaTime;
		if(m_curCycleTime > Cycles[m_curCycleIndex].Duration)
		{
			UE_LOG(LogDeepDriveTrafficLightsCtrlCmp, Log, TEXT(">> New Cycle %f"), m_curCycleTime);

			for (auto &circuit : Cycles[m_curCycleIndex].Circuits)
			{
				circuit.resetState();
			}

			m_curCycleTime = m_curCycleTime - Cycles[m_curCycleIndex].Duration;
			m_curCycleIndex = (m_curCycleIndex + 1) % Cycles.Num();

			// UE_LOG(LogDeepDriveTrafficLightsCtrlCmp, Log, TEXT("State %d"), Cycles[m_curCycleIndex].Circuits[0].getState());
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
					UE_LOG(LogDeepDriveTrafficLightsCtrlCmp, Log, TEXT(">> [%f] Switch to green"),  m_curCycleTime);
				}
			}
			else if(curState == 0)
			{
				if(m_curCycleTime > (circuit.Delay + circuit.Duration))
				{
					circuit.setState(1);
					for (auto &trafficLight : circuit.TrafficLights)
						trafficLight->SwitchToRed();
					UE_LOG(LogDeepDriveTrafficLightsCtrlCmp, Log, TEXT(">> [%f] Switch to red"), m_curCycleTime);
				}
			}
		}
	}
}
