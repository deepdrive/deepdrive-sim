
#include "DeepDrivePlugin.h"
#include "DeepDriveTrafficLight.h"

// Sets default values
ADeepDriveTrafficLight::ADeepDriveTrafficLight()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

}

// Called when the game starts or when spawned
void ADeepDriveTrafficLight::BeginPlay()
{
	Super::BeginPlay();
	
}

// Called every frame
void ADeepDriveTrafficLight::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
	if(m_remainingPhaseTime > 0.0f)
	{
		m_remainingPhaseTime -= DeltaTime;
		if(m_remainingPhaseTime <= 0.0f)
		{
			CurrentPhase = m_nextPhase;
			OnPhaseChanged();
			m_remainingPhaseTime = -1.0f;
		}
	}
}

void ADeepDriveTrafficLight::SwitchToGreen()
{
	if(RedToGreenDuration > 0.0f)
	{
		m_remainingPhaseTime = RedToGreenDuration;
		CurrentPhase = EDeepDriveTrafficLightPhase::RED_TO_GREEN;
		m_nextPhase = EDeepDriveTrafficLightPhase::GREEN;
	}
	else
	{
		m_remainingPhaseTime = -1.0f;
		CurrentPhase = EDeepDriveTrafficLightPhase::GREEN;
	}
	OnPhaseChanged();
}

void ADeepDriveTrafficLight::SwitchToRed()
{
	if(GreenToRedDuration > 0.0f)
	{
		m_remainingPhaseTime = GreenToRedDuration;
		CurrentPhase = EDeepDriveTrafficLightPhase::GREEN_TO_RED;
		m_nextPhase = EDeepDriveTrafficLightPhase::RED;
	}
	else
	{
		m_remainingPhaseTime = -1.0f;
		CurrentPhase = EDeepDriveTrafficLightPhase::RED;
	}
	OnPhaseChanged();
}

void ADeepDriveTrafficLight::SetToGreen(float ElapsedTime)
{
	if(ElapsedTime < 0.0f || ElapsedTime >= RedToGreenDuration)
		CurrentPhase = EDeepDriveTrafficLightPhase::GREEN;
	else
	{
		m_remainingPhaseTime = RedToGreenDuration - ElapsedTime;
		CurrentPhase = EDeepDriveTrafficLightPhase::RED_TO_GREEN;
		m_nextPhase = EDeepDriveTrafficLightPhase::GREEN;
	}

	OnPhaseChanged();
}

void ADeepDriveTrafficLight::SetToRed(float ElapsedTime)
{
	if(ElapsedTime < 0.0f || ElapsedTime >= GreenToRedDuration)
	{
		CurrentPhase = EDeepDriveTrafficLightPhase::RED;
		m_remainingPhaseTime = -1.0f;
	}
	else
	{
		m_remainingPhaseTime = RedToGreenDuration - ElapsedTime;
		CurrentPhase = EDeepDriveTrafficLightPhase::GREEN_TO_RED;
		m_nextPhase = EDeepDriveTrafficLightPhase::RED;
	}

	OnPhaseChanged();
}
