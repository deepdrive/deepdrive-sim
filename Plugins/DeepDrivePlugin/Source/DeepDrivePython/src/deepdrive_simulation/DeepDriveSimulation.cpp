
#include "deepdrive_simulation/DeepDriveSimulation.hpp"
#include "deepdrive_simulation/PySimulationGraphicsSettingsObject.h"
#include "deepdrive_client/DeepDriveClient.hpp"
#include "Public/Server/Messages/DeepDriveServerConfigurationMessages.h"

#include "socket/IP4ClientSocket.hpp"

#include <iostream>

DeepDriveSimulation::DeepDriveSimulation()
{
}

DeepDriveSimulation::~DeepDriveSimulation()
{
}

int32 DeepDriveSimulation::initialize(DeepDriveClient &client, uint32 seed, float timeDilation, float startLocation, PySimulationGraphicsSettingsObject &graphicsSettings)
{
	IP4ClientSocket &socket = client.getSocket();

	deepdrive::server::ConfigureSimulationRequest	req(client.getClientId(), seed, timeDilation, startLocation);
	deepdrive::server::SimulationGraphicsSettings &settings = req.graphics_settings;

	settings.is_fullscreen = graphicsSettings.is_fullscreen > 0 ? true : false;
	settings.vsync_enabled = graphicsSettings.vsync_enabled > 0 ? true : false;

	settings.resolution_width = graphicsSettings.resolution_width;
	settings.resolution_height = graphicsSettings.resolution_height;
	settings.resolution_scale = graphicsSettings.resolution_scale;

	settings.texture_quality = graphicsSettings.texture_quality;
	settings.shadow_quality = graphicsSettings.shadow_quality;
	settings.effect_quality = graphicsSettings.effect_quality;
	settings.post_process_level = graphicsSettings.post_process_level;
	settings.motion_blur_quality = graphicsSettings.motion_blur_quality;
	settings.view_distance = graphicsSettings.view_distance;
	settings.ambient_occlusion = graphicsSettings.ambient_occlusion;

	int32 res = socket.send(&req, sizeof(req));
	if(res >= 0)
	{
		std::cout << "ConfigureSimulationRequest sent\n";

		deepdrive::server::ConfigureSimulationResponse response;
		if(socket.receive(&response, sizeof(response), 1000))
		{
			res = static_cast<int32> (response.initialized);
			std::cout << "InitializeSimulationResponse received " << client.getClientId() << " " << res << "\n";
		}
		else
			std::cout << "Waiting for ConfigureSimulationRequest, time out\n";
	}

	return res;
}


int32 DeepDriveSimulation::setSunSimulation(DeepDriveClient &client, uint32 month, uint32 day, uint32 minute, uint32 hour, uint32 speed)
{
	deepdrive::server::SetSunSimulationRequest	req(client.getClientId(), month, day, hour, minute, speed);
	IP4ClientSocket &socket = client.getSocket();

	int32 res = socket.send(&req, sizeof(req));
	if(res >= 0)
	{
		std::cout << "SetSunSimulationRequest sent\n";

		deepdrive::server::SetSunSimulationResponse response;
		if(socket.receive(&response, sizeof(response), 1000))
		{
			res = static_cast<int32> (response.result);
			std::cout << "SetSunSimulationResponse received " << client.getClientId() << " " << res << "\n";
		}
		else
			std::cout << "Waiting for SetSunSimulationRequest, time out\n";
	}

	return res;
}
