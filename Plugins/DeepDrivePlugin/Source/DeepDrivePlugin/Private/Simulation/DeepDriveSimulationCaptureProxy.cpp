

#include "DeepDrivePluginPrivatePCH.h"
#include "Public/Simulation/DeepDriveSimulationCaptureProxy.h"
#include "Public/Simulation/DeepDriveSimulation.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"


#include "Private/Capture/DeepDriveCapture.h"

DEFINE_LOG_CATEGORY(LogDeepDriveSimulationCaptureProxy);

DeepDriveSimulationCaptureProxy::DeepDriveSimulationCaptureProxy(ADeepDriveSimulation &deepDriveSim, float captureInterval)
	:	m_DeepDriveSim(deepDriveSim)
	,	m_CaptureInterval(captureInterval)
{
	DeepDriveCapture::GetInstance().RegisterProxy(*this);
	UE_LOG(LogDeepDriveSimulationCaptureProxy, Log, TEXT("Capture Proxy registered"));
}

void DeepDriveSimulationCaptureProxy::update( float DeltaSeconds )
{
	DeepDriveCapture &deepDriveCapture = DeepDriveCapture::GetInstance();

	deepDriveCapture.HandleCaptureResult();

	if(m_CaptureInterval >= 0.0f)
	{
		m_TimeToNextCapture -= DeltaSeconds;

		if(m_TimeToNextCapture <= 0.0f)
		{
			if(beginCapture())
				DeepDriveCapture::GetInstance().Capture();

			m_TimeToNextCapture = m_CaptureInterval;
		}
	}
}

void DeepDriveSimulationCaptureProxy::shutdown()
{
	DeepDriveCapture::GetInstance().UnregisterProxy(*this);
	UE_LOG(LogDeepDriveSimulationCaptureProxy, Log, TEXT("Capture Proxy unregistered"));
}

bool DeepDriveSimulationCaptureProxy::beginCapture()
{
	ADeepDriveAgent *agent = m_DeepDriveSim.getCurrentAgent();

	if(agent)
	{
		#if 0
		m_DeepDriveData.Position = agent->GetActorLocation();
		m_DeepDriveData.Rotation = agent->GetActorRotation();
		m_DeepDriveData.Velocity = agent->GetVelocity();
		m_DeepDriveData.AngularVelocity = agent->getAngularVelocity();
		m_DeepDriveData.Acceleration = agent->getAcceleration();
		m_DeepDriveData.AngularAcceleration = agent->getAngularAcceleration();
		m_DeepDriveData.Speed = agent->getSpeed();

		m_DeepDriveData.Dimension = agent->getDimensions();

		m_DeepDriveData.IsGameDriving = agent->getIsGameDriving();

		m_DeepDriveData.Steering = agent->getSteering();
		m_DeepDriveData.Throttle = agent->getThrottle();
		m_DeepDriveData.Brake = agent->getBrake();
		m_DeepDriveData.Handbrake = agent->getHandbrake();

		m_DeepDriveData.DistanceAlongRoute = agent->getDistanceAlongRoute();
		m_DeepDriveData.DistanceToCenterOfLane = agent->getDistanceToCenterOfTrack();
		m_DeepDriveData.LapNumber = agent->getNumberOfLaps();

		agent->getLastCollisionTime(m_DeepDriveData.LastCollisionTimeUTC, m_DeepDriveData.LastCollisionTimeStamp, m_DeepDriveData.TimeSinceLastCollision);
		//m_DeepDriveData.LastCollisionTime = agent->getLastCollisionTime();
		#endif

		agent->beginCapture(m_DeepDriveData);
	}

	return agent != 0;
}

/**
*		IDeepDriveCaptureProxy methods
*/

TArray<UCaptureSinkComponentBase*>& DeepDriveSimulationCaptureProxy::getSinks()
{
	return m_DeepDriveSim.getCaptureSinks();
}
