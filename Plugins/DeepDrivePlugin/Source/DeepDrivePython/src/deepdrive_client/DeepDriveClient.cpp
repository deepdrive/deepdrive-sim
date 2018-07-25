
#include "deepdrive_client/DeepDriveClient.hpp"
#include "deepdrive_simulation/PySimulationGraphicsSettingsObject.h"
#include "common/ClientErrorCode.hpp"

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

int32 DeepDriveClient::registerClient	(	deepdrive::server::RegisterClientResponse &response, bool requestMasterRole
										,	uint32 seed, float timeDilation, float startLocation
										,	PySimulationGraphicsSettingsObject *graphicsSettings
										)
{
	uint32 clientId = 0;

	deepdrive::server::RegisterClientRequest req(requestMasterRole);

	SimulationConfiguration &cfg = req.configuration;
	cfg.seed = seed;
	cfg.time_dilation = timeDilation;
	cfg.agent_start_location = startLocation;

	if(graphicsSettings)
	{
		SimulationGraphicsSettings &gfxSettings = req.graphics_settings;

		gfxSettings.is_fullscreen = graphicsSettings->is_fullscreen;
		gfxSettings.vsync_enabled = graphicsSettings->vsync_enabled;
		gfxSettings.resolution_width = graphicsSettings->resolution_width;
		gfxSettings.resolution_height = graphicsSettings->resolution_height;
		gfxSettings.resolution_scale = graphicsSettings->resolution_scale;
		gfxSettings.texture_quality = static_cast<uint8> (graphicsSettings->texture_quality);
		gfxSettings.shadow_quality = static_cast<uint8> (graphicsSettings->shadow_quality);
		gfxSettings.effect_quality = static_cast<uint8> (graphicsSettings->effect_quality);
		gfxSettings.post_process_level = static_cast<uint8> (graphicsSettings->post_process_level);
		gfxSettings.motion_blur_quality = static_cast<uint8> (graphicsSettings->motion_blur_quality);
		gfxSettings.view_distance = static_cast<uint8> (graphicsSettings->view_distance);
		gfxSettings.ambient_occlusion = static_cast<uint8> (graphicsSettings->ambient_occlusion);
	}

	int32 res = m_Socket.send(&req, sizeof(req));
	if(res >= 0)
	{
		std::cout << "RegisterClientRequest sent\n";

		res = m_Socket.receive(&response, sizeof(response));

		if(res > 0)
		{
			res = ClientErrorCode::NO_ERROR;
			clientId = m_ClientId = response.client_id;
			m_isMaster = response.granted_master_role > 0 ? true : false;

			m_ServerProtocolVersion = response.server_protocol_version;

			m_SharedMemoryName = std::string(response.shared_memory_name);
			m_SharedMemorySize = response.shared_memory_size;

			m_MaxSupportedCameras = response.max_supported_cameras;
			m_MaxCaptureResolution = response.max_capture_resolution;

			m_InactivityTimeout = response.inactivity_timeout_ms;

			std::cout << "RegisterClientResponse received client id " << m_ClientId << " max cams "
			 	<< m_MaxSupportedCameras << " capture res " << m_MaxCaptureResolution <<  " protocol version " << m_ServerProtocolVersion << "\n";
		}
	}

	return res;
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


int32 DeepDriveClient::registerCamera(float hFoV, uint16 captureWidth, uint16 captureHeight, float relPos[3], float relRot[3], const char *label)
{
	deepdrive::server::RegisterCaptureCameraRequest req(m_ClientId, hFoV, captureWidth, captureHeight, label);
	req.relative_position[0] = relPos[0];	req.relative_position[1] = relPos[1];	req.relative_position[2] = relPos[2];
	req.relative_rotation[0] = relRot[0];	req.relative_rotation[1] = relRot[1];	req.relative_rotation[2] = relRot[2];

	int32 res = m_Socket.send(&req, sizeof(req));
	if(res >= 0)
	{
		std::cout << "RegisterCaptureCameraRequest sent\n";

		deepdrive::server::RegisterCaptureCameraResponse response;
		if(m_Socket.receive(&response, sizeof(response), 1000))
		{
			res = static_cast<int32> (response.camera_id);
			std::cout << "RegisterCaptureCameraResponse received " << m_ClientId << " " << res << "\n";
		}
		else
			std::cout << "Waiting for RegisterCaptureCameraResponse, time out\n";
	}

	return res;
}

int32 DeepDriveClient::requestAgentControl()
{
	int32 res = ClientErrorCode::NOT_CONNECTED;
	deepdrive::server::RequestAgentControlRequest req(m_ClientId);
	res =m_Socket.send(&req, sizeof(req));

	if(res >= 0)
	{
	//	std::cout << "RequestAgentControlRequest sent\n";

		deepdrive::server::RequestAgentControlResponse response;
		if(m_Socket.receive(&response, sizeof(response), 1000))
		{
			res = response.control_granted? 1 : 0;
	//		std::cout << "RequestAgentControlResponse received " << m_ClientId << " " << response.control_granted << "\n";
		}
		else
			std::cout << "Waiting for RequestAgentControlResponse, time out\n";
	}

	return res;
}

int32 DeepDriveClient::releaseAgentControl()
{
	int32 res = ClientErrorCode::NOT_CONNECTED;
	deepdrive::server::ReleaseAgentControlRequest req(m_ClientId);
	res =m_Socket.send(&req, sizeof(req));
	if(res >= 0)
	{
	//	std::cout << "ReleaseAgentControlRequest sent\n";

		deepdrive::server::ReleaseAgentControlResponse response;
		if(m_Socket.receive(&response, sizeof(response), 1000))
		{
	//		std::cout << "ReleaseAgentControlResponse received " << m_ClientId << "\n";
		}
		else
			std::cout << "Waiting for ReleaseAgentControlResponse, time out\n";
	}

	return res;
}

int32 DeepDriveClient::resetAgent()
{
	int32 res = ClientErrorCode::NOT_CONNECTED;

	deepdrive::server::ResetAgentRequest req(m_ClientId);
	res = m_Socket.send(&req, sizeof(req));
	if(res >= 0)
	{
		std::cout << "ResetAgentRequest sent " << m_ClientId << "\n";

		deepdrive::server::ResetAgentResponse response;
		if(m_Socket.receive(&response, sizeof(response), 2500))
		{
			std::cout << "ResetAgentResponse received " << m_ClientId << "\n";
		}
		else
			std::cout << "Waiting for ResetAgentResponse, time out\n";
	}

	return res;
}

int32 DeepDriveClient::setControlValues(float steering, float throttle, float brake, uint32 handbrake)
{
	int32 res = ClientErrorCode::NOT_CONNECTED;
	if(m_Socket.isConnected())
	{
		deepdrive::server::SetAgentControlValuesRequest req(m_ClientId, steering, throttle, brake, handbrake);
		res = m_Socket.send(&req, sizeof(req));
	}
	return res;
}

int32 DeepDriveClient::activateSynchronousStepping()
{
	int32 res = ClientErrorCode::NOT_CONNECTED;
	deepdrive::server::ActivateSynchronousSteppingRequest req(m_ClientId);
	res = m_Socket.send(&req, sizeof(req));

	if(res >= 0)
	{
	//	std::cout << "ActivateSynchronousSteppingRequest sent\n";

		deepdrive::server::ActivateSynchronousSteppingResponse response;
		if(m_Socket.receive(&response, sizeof(response), 1000))
		{
			res = response.synchronous_stepping_activated ? 1 : 0;
			std::cout << "ActivateSynchronousSteppingResponse received " << m_ClientId << " " << response.synchronous_stepping_activated << "\n";
		}
		else
			std::cout << "Waiting for ActivateSynchronousSteppingResponse, time out\n";
	}

	return res;
}

int32 DeepDriveClient::deactivateSynchronousStepping()
{
	int32 res = ClientErrorCode::NOT_CONNECTED;
	deepdrive::server::DeactivateSynchronousSteppingRequest req(m_ClientId);
	res = m_Socket.send(&req, sizeof(req));
	if(res >= 0)
	{
	//	std::cout << "DeactivateSynchronousSteppingRequest sent\n";

		deepdrive::server::DeactivateSynchronousSteppingResponse response;
		if(m_Socket.receive(&response, sizeof(response), 1000))
		{
			std::cout << "DeactivateSynchronousSteppingResponse received " << m_ClientId << "\n";
		}
		else
			std::cout << "Waiting for DeactivateSynchronousSteppingResponse, time out\n";
	}

	return res;
}


int32 DeepDriveClient::advanceSynchronousStepping(float dT, float steering, float throttle, float brake, uint32 handbrake)
{
	int32 res = ClientErrorCode::NOT_CONNECTED;
	if(m_Socket.isConnected())
	{
		deepdrive::server::AdvanceSynchronousSteppingRequest req(m_ClientId, dT, steering, throttle, brake, handbrake);
		res = m_Socket.send(&req, sizeof(req));
		if(res >= 0)
		{
		//	std::cout << "DeactivateSynchronousSteppingRequest sent\n";

			deepdrive::server::AdvanceSynchronousSteppingResponse response;
			if(m_Socket.receive(&response, sizeof(response)))
			{
				res = response.sequence_number;
		//		std::cout << "AdvanceSynchronousSteppingResponse received " << m_ClientId << "\n";
			}
			else
				std::cout << "Waiting for AdvanceSynchronousSteppingResponse, time out\n";
		}
	}
	return res;
}

