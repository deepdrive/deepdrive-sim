

#include "DeepDrivePluginPrivatePCH.h"
#include "Public/Simulation/DeepDriveSimulationServerProxy.h"
#include "Public/Simulation/DeepDriveSimulation.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"
#include "Public/Simulation/Agent/DeepDriveAgentControllerBase.h"
#include "Public/Simulation/DeepDriveSimulationTypes.h"

#include "Private/Server/DeepDriveServer.h"

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

bool DeepDriveSimulationServerProxy::RequestAgentControl()
{
	bool res = true;

	UE_LOG(LogDeepDriveSimulationServerProxy, Log, TEXT("Control requested") );

	return res;
}

void DeepDriveSimulationServerProxy::ReleaseAgentControl()
{
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
