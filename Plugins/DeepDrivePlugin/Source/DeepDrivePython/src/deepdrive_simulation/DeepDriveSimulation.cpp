
#include "deepdrive_simulation/DeepDriveSimulation.hpp"
#include "deepdrive_simulation/PySimulationGraphicsSettingsObject.h"

#include "Public/Server/Messages/DeepDriveServerSimulationMessages.h"

#include "socket/IP4ClientSocket.hpp"
#include "common/ClientErrorCode.hpp"

#include <iostream>

DeepDriveSimulation::DeepDriveSimulation(const IP4Address &ip4Address)
	:	m_Socket()
{
	m_Socket.connect(ip4Address);
}

DeepDriveSimulation::~DeepDriveSimulation()
{
}


int32 DeepDriveSimulation::configureSimulation(uint32 seed, float timeDilation, float startLocation, PySimulationGraphicsSettingsObject *graphicsSettings)
{
	deepdrive::server::ConfigureSimulationRequest req;

	SimulationConfiguration &cfg = req.configuration;
	cfg.seed = 0;
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
		std::cout << "ConfigureSimulationRequest sent\n";

		deepdrive::server::ConfigureSimulationResponse response;
		if(m_Socket.receive(&response, sizeof(response), 1000))
		{
			res = static_cast<int32> (response.success);
			std::cout << "ConfigureSimulationResponse received \n";
		}
		else
		{
			res = ClientErrorCode::TIME_OUT;
			std::cout << "Waiting for ConfigureSimulationResponse, time out\n";
		}
	}

	return res;
}

int32 DeepDriveSimulation::resetSimulation(float timeDilation, float startLocation, PySimulationGraphicsSettingsObject *graphicsSettings)
{
#if 0
	deepdrive::server::ResetSimulationRequest req(client.getClientId());

	SimulationConfiguration &cfg = req.configuration;
	cfg.seed = 0;
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


	IP4ClientSocket &socket = client.getSocket();

	int32 res = socket.send(&req, sizeof(req));
	if(res >= 0)
	{
		std::cout << "ResetSimulationRequest sent\n";

		deepdrive::server::ResetSimulationResponse response;
		if(socket.receive(&response, sizeof(response), 1000))
		{
			res = static_cast<int32> (response.reset);
			std::cout << "ResetSimulationResponse received " << client.getClientId() << " " << res << "\n";
		}
		else
			std::cout << "Waiting for ResetSimulationResponse, time out\n";
	}

	return res;
#endif

	return 0;
}

int32 DeepDriveSimulation::setDateAndTime(uint32 year, uint32 month, uint32 day, uint32 minute, uint32 hour)
{
	deepdrive::server::SetDateAndTimeRequest	req(year, month, day, hour, minute);

	int32 res = m_Socket.send(&req, sizeof(req));
	if(res >= 0)
	{
		std::cout << "SetDateAndTimeRequest sent\n";

		deepdrive::server::SetDateAndTimeResponse response;
		if(m_Socket.receive(&response, sizeof(response), 1000))
		{
			res = static_cast<int32> (response.result);
			std::cout << "SetDateAndTimeResponse received\n";
		}
		else
		{
			std::cout << "Waiting for SetDateAndTimeResponse, time out\n";
			res = TIME_OUT;
		}
	}

	return res;
}

int32 DeepDriveSimulation::setSpeed(uint32 speed)
{
	deepdrive::server::SetSunSimulationSpeedRequest	req(speed);

	int32 res = m_Socket.send(&req, sizeof(req));
	if(res >= 0)
	{
		std::cout << "SetSunSimulationSpeedRequest sent\n";

		deepdrive::server::SetSunSimulationSpeedResponse response;
		if(m_Socket.receive(&response, sizeof(response), 1000))
		{
			res = static_cast<int32> (response.result);
			std::cout << "SetSunSimulationSpeedResponse received\n";
		}
		else
		{
			std::cout << "Waiting for SetSunSimulationSpeedResponse, time out\n";
			res = TIME_OUT;
		}
	}

	return res;
}

