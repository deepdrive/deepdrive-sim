

#include "DeepDrivePluginPrivatePCH.h"
#include "DeepDriveAgentSplineController.h"

DEFINE_LOG_CATEGORY(LogDeepDriveAgentSplineController);

bool ADeepDriveAgentSplineController::Activate(ADeepDriveAgent &agent)
{
	/*
		A bit of a hack. Find spline to run on by searching for actors tagged with AgentSpline and having a spline component
	*/

	TArray<AActor*> actors;
	UGameplayStatics::GetAllActorsWithTag(GetWorld(), TEXT("AgentSpline"), actors);

	for(auto actor : actors)
	{
		TArray <UActorComponent*> splines = actor->GetComponentsByClass( USplineComponent::StaticClass() );
		if(splines.Num() > 0)
		{
			m_Spline = Cast<USplineComponent> (splines[0]);
			if(m_Spline)
				break;
		}
	}

	return m_Spline != 0 && Super::Activate(agent);
}

void ADeepDriveAgentSplineController::Tick( float DeltaSeconds )
{

}
