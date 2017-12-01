
#include "deepdrive_Client/DeepDriveClient.hpp"

#include "Public/Server/Messages/DeepDriveServerConnectionMessages.h"
#include "Public/Server/Messages/DeepDriveServerConfigurationMessages.h"
#include "Public/Server/Messages/DeepDriveServerControlMessages.h"

#include <iostream>

DeepDriveClient::DeepDriveClient(const IP4Address &ip4Address)
	:	m_Socket()
{
	m_Socket.connect(ip4Address);
}


DeepDriveClient::~DeepDriveClient()
{

}

uint32 DeepDriveClient::registerClient()
{
	uint32 clientId = 0;

	deepdrive::server::RegisterClientRequest req(true);
	m_Socket.send(&req, sizeof(req));

	std::cout << "RegisterClientRequest sent\n";

	deepdrive::server::RegisterClientResponse response;
	if(m_Socket.receive(&response, sizeof(response), 2000))
	{
		clientId = m_ClientId = response.client_id;
		m_isMaster = response.granted_master_role;

		m_ServerProtocolVersion = response.server_protocol_version;

		m_SharedMemoryName = std::string(response.shared_memory_name);
		m_SharedMemorySize = response.shared_memory_size;

		m_MaxSupportedCameras = response.max_supported_cameras;
		m_MaxCaptureResolution = response.max_capture_resolution;

		m_InactivityTimeout = response.inactivity_timeout_ms;

		std::cout << "RegisterClientResponse received " << m_ClientId << " " << m_MaxSupportedCameras << " " << m_MaxCaptureResolution << "\n";
	}
	else
		std::cout << "Waiting for RegisterClientResponse, time out\n";

	return clientId;
}

bool DeepDriveClient::isConnected() const
{
	return m_Socket.isConnected();
}


void DeepDriveClient::close()
{
	deepdrive::server::UnregisterClientRequest req(m_ClientId);
	m_Socket.send(&req, sizeof(req));

	std::cout << "UnregisterClientRequest sent\n";

	deepdrive::server::UnregisterClientResponse response;
	if(m_Socket.receive(&response, sizeof(response), 1000))
		std::cout << "Successfully unregistered\n";
	m_Socket.close();
}


uint32 DeepDriveClient::registerCamera(float hFoV, uint16 captureWidth, uint16 captureHeight, float relPos[3], float relRot[3])
{
	uint32 camId = 0;
	deepdrive::server::RegisterCaptureCameraRequest req(m_ClientId, hFoV, captureWidth, captureHeight);
	req.relative_position[0] = relPos[0];	req.relative_position[1] = relPos[1];	req.relative_position[2] = relPos[2];
	req.relative_rotation[0] = relRot[0];	req.relative_rotation[1] = relRot[1];	req.relative_rotation[2] = relRot[2];

	m_Socket.send(&req, sizeof(req));

	std::cout << "RegisterCaptureCameraRequest sent\n";

	deepdrive::server::RegisterCaptureCameraResponse response;
	if(m_Socket.receive(&response, sizeof(response), 1000))
	{
		camId = response.camera_id;
		std::cout << "RegisterCaptureCameraResponse received " << m_ClientId << " " << camId << "\n";
	}
	else
		std::cout << "Waiting for RegisterCaptureCameraResponse, time out\n";

	return camId;
}

bool DeepDriveClient::requestAgentControl()
{
	bool ctrlGranted = false;
	deepdrive::server::RequestAgentControlRequest req(m_ClientId);
	m_Socket.send(&req, sizeof(req));

	std::cout << "RequestAgentControlRequest sent\n";

	deepdrive::server::RequestAgentControlResponse response;
	if(m_Socket.receive(&response, sizeof(response), 1000))
	{
		ctrlGranted = response.control_granted;
		std::cout << "RequestAgentControlResponse received " << m_ClientId << " " << ctrlGranted << "\n";
	}
	else
		std::cout << "Waiting for RequestAgentControlResponse, time out\n";

	return ctrlGranted;
}

void DeepDriveClient::releaseAgentControl()
{
	deepdrive::server::ReleaseAgentControlRequest req(m_ClientId);
	m_Socket.send(&req, sizeof(req));

	std::cout << "ReleaseAgentControlRequest sent\n";

	deepdrive::server::ReleaseAgentControlResponse response;
	if(m_Socket.receive(&response, sizeof(response), 1000))
	{
		std::cout << "ReleaseAgentControlResponse received " << m_ClientId << "\n";
	}
	else
		std::cout << "Waiting for ReleaseAgentControlResponse, time out\n";
}


void DeepDriveClient::setControlValues(float steering, float throttle, float brake, uint32 handbrake)
{
	deepdrive::server::SetAgentControlValuesRequest req(m_ClientId, steering, throttle, brake, handbrake);
	m_Socket.send(&req, sizeof(req));

	std::cout << "SetAgentControlValueRequest sent\n";
}
