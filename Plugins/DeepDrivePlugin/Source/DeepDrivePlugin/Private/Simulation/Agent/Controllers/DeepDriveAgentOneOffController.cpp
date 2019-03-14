

#include "DeepDrivePluginPrivatePCH.h"
#include "DeepDriveAgentOneOffController.h"
#include "Simulation/Misc/DeepDriveSplineTrack.h"
#include "Runtime/Engine/Classes/Components/SplineComponent.h"


ADeepDriveAgentOneOffController::ADeepDriveAgentOneOffController()
	:	ADeepDriveAgentControllerBase()
{
	m_ControllerName = "OneOff Controller";

	m_isCollisionEnabled = true;
}

bool ADeepDriveAgentOneOffController::Activate(ADeepDriveAgent &agent, bool keepPosition)
{
	bool res = keepPosition;

	if (!res && m_bindToRoad)
	{
		TArray<AActor*> actors;
		UGameplayStatics::GetAllActorsOfClass(GetWorld(), ADeepDriveSplineTrack::StaticClass(), actors);
		float bestDist = TNumericLimits<float>::Max();
		ADeepDriveSplineTrack *closestTrack = 0;
		float closestKey = -1.0f;
		FVector startLocation = m_StartTransform.GetLocation();
		for (auto &actor : actors)
		{
			ADeepDriveSplineTrack *track = Cast<ADeepDriveSplineTrack>(actor);
			if (track)
			{
				USplineComponent *spline = track->GetSpline();
				if (spline)
				{
					const float curKey = spline->FindInputKeyClosestToWorldLocation(startLocation);
					const float curDist = (spline->GetLocationAtSplineInputKey(curKey, ESplineCoordinateSpace::World) - startLocation).Size();
					if (curDist < bestDist)
					{
						bestDist = curDist;
						closestTrack = track;
						closestKey = curKey;
					}
				}
			}
		}

		if (closestTrack)
		{
			m_StartDistance = getClosestDistanceOnSpline(closestTrack->GetSpline(), startLocation);
			m_Track = closestTrack;
			res = initAgentOnTrack(agent);
		}
	}

	if(res)
	{
		activateController(agent);
	}
	return res;
}
