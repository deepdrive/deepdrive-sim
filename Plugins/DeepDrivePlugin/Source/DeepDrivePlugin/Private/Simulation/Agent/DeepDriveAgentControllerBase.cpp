

#include "DeepDrivePluginPrivatePCH.h"

#include "Public/Simulation/DeepDriveSimulation.h"
#include "Public/Simulation/Agent/DeepDriveAgentControllerBase.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"
#include "Public/Simulation/Misc/DeepDriveSplineTrack.h"
#include "Runtime/Engine/Classes/Components/SplineComponent.h"

DEFINE_LOG_CATEGORY(LogDeepDriveAgentControllerBase);

ADeepDriveAgentControllerBase::ADeepDriveAgentControllerBase()
{
}

ADeepDriveAgentControllerBase::~ADeepDriveAgentControllerBase()
{
	UE_LOG(LogDeepDriveAgentControllerBase, Log, TEXT("~ADeepDriveAgentControllerBase: %p sayz bye"), this );
}

void ADeepDriveAgentControllerBase::OnConfigureSimulation(const SimulationConfiguration &configuration, bool initialConfiguration)
{
}

bool ADeepDriveAgentControllerBase::Activate(ADeepDriveAgent &agent, bool keepPosition)
{
	return false;
}

void ADeepDriveAgentControllerBase::Deactivate()
{
}

void ADeepDriveAgentControllerBase::MoveForward(float axisValue)
{

}

void ADeepDriveAgentControllerBase::MoveRight(float axisValue)
{

}	

void ADeepDriveAgentControllerBase::SetControlValues(float steering, float throttle, float brake, bool handbrake)
{

}

bool ADeepDriveAgentControllerBase::ResetAgent()
{
	bool res = false;
	if(m_Agent)
	{
		m_Agent->reset();
		res = true;
	}
	return res;
}

void ADeepDriveAgentControllerBase::OnAgentCollision(AActor *OtherActor, const FHitResult &HitResult, const FName &Tag)
{
	if (m_isCollisionEnabled)
	{
		FDateTime now(FDateTime::UtcNow());
		m_CollisionData.LastCollisionTimeStamp = FPlatformTime::Seconds();
		m_CollisionData.LastCollisionTimeUTC = FDateTime::UtcNow();
		m_CollisionData.CollisionLocation = Tag;
		m_CollisionData.ColliderVelocity = OtherActor->GetVelocity();
		m_CollisionData.CollisionNormal = HitResult.ImpactNormal;
		m_hasCollisionOccured = true;

		UE_LOG(LogDeepDriveAgentControllerBase, Log, TEXT("ADeepDriveAgentControllerBase::OnAgentCollision %ld"), now.ToUnixTimestamp() );

	}
}


void ADeepDriveAgentControllerBase::activateController(ADeepDriveAgent &agent)
{
	m_Agent = &agent;
	agent.setAgentController(this);
	Possess(m_Agent);
	agent.setIsGameDriving(m_isGameDriving);
}

bool ADeepDriveAgentControllerBase::initAgentOnTrack(ADeepDriveAgent &agent)
{
	bool res = false;
	if(m_Track && m_DeepDriveSimulation)
	{
		m_Track->registerAgent(agent, m_Track->GetSpline()->FindInputKeyClosestToWorldLocation(agent.GetActorLocation()));
		if(m_StartDistance < 0.0f)
		{
			m_StartDistance = m_Track->getRandomDistanceAlongTrack(*m_DeepDriveSimulation->GetRandomStream(FName("AgentPlacement")));
			UE_LOG(LogDeepDriveAgentControllerBase, Log, TEXT("ADeepDriveAgentControllerBase::initAgentOnTrack random start distance %f"), m_StartDistance);
		}

		if(m_StartDistance >= 0.0f)
		{
			resetAgentPosOnSpline(agent, m_Track->GetSpline(), m_StartDistance);
			res = true;
			UE_LOG(LogDeepDriveAgentControllerBase, Log, TEXT("ADeepDriveAgentControllerBase::initAgentOnTrack Successfully initialized"));
		}
	}
	return res;
}

bool ADeepDriveAgentControllerBase::updateAgentOnTrack()
{
	bool res = false;

	if(m_Track)
	{
		const float curDistanceOnSpline = getClosestDistanceOnSpline(m_Track->GetSpline(), m_Agent->GetActorLocation());

		if(m_LapStarted)
		{
			const float delta = curDistanceOnSpline - m_StartDistance;
			res = delta > 0.0f && delta < m_LapDistanceThreshold;
			if(res)
				m_LapStarted = false;
		}
		else if(curDistanceOnSpline > (m_StartDistance + m_LapDistanceThreshold))
		{
			m_LapStarted = true;
		}
	}

	return res;
}

void ADeepDriveAgentControllerBase::resetAgentPosOnSpline(ADeepDriveAgent &agent, USplineComponent *spline, float distance)
{
	FVector agentLocation = spline->GetLocationAtDistanceAlongSpline(distance, ESplineCoordinateSpace::World) + FVector(0.0f, 0.0f, 200.0f);
	float curDistanceOnSpline = getClosestDistanceOnSpline(spline, agentLocation);
	FVector curPosOnSpline = spline->GetLocationAtDistanceAlongSpline(curDistanceOnSpline, ESplineCoordinateSpace::World);
	curPosOnSpline.Z = agentLocation.Z + 50.0f;

	FQuat quat = spline->GetQuaternionAtDistanceAlongSpline(curDistanceOnSpline, ESplineCoordinateSpace::World);

	FTransform transform(quat.Rotator(), curPosOnSpline, FVector(1.0f, 1.0f, 1.0f));

	agent.SetActorTransform(transform, false, 0, ETeleportType::TeleportPhysics);
}

float ADeepDriveAgentControllerBase::getClosestDistanceOnSpline(USplineComponent *spline, const FVector &location)
{
	float distance = 0.0f;

	const float closestKey = spline->FindInputKeyClosestToWorldLocation(location);

	const int32 index0 = floor(closestKey);
	const int32 index1 = floor(closestKey + 1.0f);

	const float dist0 = spline->GetDistanceAlongSplineAtSplinePoint(index0);
	const float dist1 = spline->GetDistanceAlongSplineAtSplinePoint(index1);


	return FMath::Lerp(dist0, dist1, closestKey - static_cast<float> (index0));
}

void ADeepDriveAgentControllerBase::getCollisionData(DeepDriveCollisionData &collisionDataOut)
{
	collisionDataOut = m_hasCollisionOccured ? m_CollisionData : DeepDriveCollisionData();
}


void ADeepDriveAgentControllerBase::OnCheckpointReached()
{
}

void ADeepDriveAgentControllerBase::OnDebugTrigger()
{
}

