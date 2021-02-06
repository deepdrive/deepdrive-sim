

#include "Simulation/DeepDriveSimulationServerProxy.h"
#include "Simulation/DeepDriveSimulation.h"
#include "Simulation/DeepDriveSimulationDefines.h"
#include "Simulation/Agent/DeepDriveAgent.h"
#include "Simulation/Agent/DeepDriveAgentControllerBase.h"
#include "Simulation/DeepDriveSimulationTypes.h"

#include "Server/DeepDriveServer.h"

DEFINE_LOG_CATEGORY(LogDeepDriveSimulationServerProxy);

DeepDriveSimulationServerProxy::DeepDriveSimulationServerProxy(ADeepDriveSimulation &deepDriveSim)
	:	m_DeepDriveSim(deepDriveSim)
	,	m_isActive(false)
{
}

bool DeepDriveSimulationServerProxy::initialize(const FString &clientIPAddress, int32 clientPort, UWorld *world)
{
	m_isActive = false;
	if	(	clientPort > 0 && clientPort <= 65535
		&&	DeepDriveServer::GetInstance().RegisterProxy(*this, clientIPAddress, static_cast<uint16> (clientPort))
		)
	{
		DeepDriveServer::GetInstance().setWorld(world);
		m_isActive = true;
		UE_LOG(LogDeepDriveSimulationServerProxy, Log, TEXT("Server Proxy registered"));
	}
	else
	{
		UE_LOG(LogDeepDriveSimulationServerProxy, Error, TEXT("Server Proxy could not be registered"));
	}

	return m_isActive;
}

void DeepDriveSimulationServerProxy::update( float DeltaSeconds )
{
	if(m_isActive)
		DeepDriveServer::GetInstance().update(DeltaSeconds);
}

void DeepDriveSimulationServerProxy::shutdown()
{
	if(m_isActive)
	{
		DeepDriveServer::GetInstance().UnregisterProxy(*this);
		UE_LOG(LogDeepDriveSimulationServerProxy, Log, TEXT("Server Proxy unregistered"));
	}
}

/**
*		IDeepDriveServerProxy methods
*/

void DeepDriveSimulationServerProxy::RegisterClient(int32 ClientId, bool IsMaster)
{
	UE_LOG(LogDeepDriveSimulationServerProxy, Log, TEXT("Client [%d] registered isMaster %c"), ClientId, IsMaster ? 'T' : 'F');
}

void DeepDriveSimulationServerProxy::UnregisterClient(int32 ClientId, bool IsMaster)
{
	UE_LOG(LogDeepDriveSimulationServerProxy, Log, TEXT("Client [%d] unregistered"), ClientId);
}

int32 DeepDriveSimulationServerProxy::RegisterCaptureCamera(float FieldOfView, int32 CaptureWidth, int32 CaptureHeight, FVector RelativePosition, FVector RelativeRotation, const FString &Label)
{
	ADeepDriveAgent *agent = m_isActive ? m_DeepDriveSim.getCurrentAgent() : 0;
	return agent ? agent->RegisterCaptureCamera(FieldOfView, CaptureWidth, CaptureHeight, RelativePosition, RelativeRotation, Label) : 0;
}

void DeepDriveSimulationServerProxy::UnregisterCaptureCamera(uint32 cameraId)
{
	ADeepDriveAgent *agent = m_isActive ? m_DeepDriveSim.getCurrentAgent() : 0;
	if(agent)
		agent->UnregisterCaptureCamera(cameraId);
}

bool DeepDriveSimulationServerProxy::RequestAgentControl()
{
	bool res = m_DeepDriveSim.requestControl();

	UE_LOG(LogDeepDriveSimulationServerProxy, Log, TEXT("Control requested - ensuring agent controller is RemoteAI"));
	return res;
}

void DeepDriveSimulationServerProxy::ReleaseAgentControl()
{
	m_DeepDriveSim.releaseControl();

	UE_LOG(LogDeepDriveSimulationServerProxy, Log, TEXT("Control released"));
}

void DeepDriveSimulationServerProxy::ResetAgent()
{
	bool res = m_isActive ? m_DeepDriveSim.resetAgent() : false;
	DeepDriveServer::GetInstance().onAgentReset(res);
}

void DeepDriveSimulationServerProxy::SetAgentControlValues(float steering, float throttle, float brake, bool handbrake)
{
	ADeepDriveAgentControllerBase *agentCtrl = m_isActive ? m_DeepDriveSim.getCurrentAgentController() : 0;

	if (agentCtrl)
	{
		agentCtrl->SetControlValues(steering, throttle, brake, handbrake);
	}
}

bool DeepDriveSimulationServerProxy::SetViewMode(int32 cameraId, const FString &viewMode)
{
	bool res = false;

	ADeepDriveAgent *agent = m_DeepDriveSim.getCurrentAgent();

	if (agent)
	{
		res = agent->setViewMode(cameraId, viewMode);
	}
	else
		UE_LOG(LogDeepDriveSimulationServerProxy, Error, TEXT("DeepDriveSimulationServerProxy::SetViewMode failed, no agent"));


	return res;
}
