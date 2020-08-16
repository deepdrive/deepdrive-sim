
#include "deepdrive_simulation/DeepDriveSimulation.hpp"
#include "deepdrive_simulation/PySimulationGraphicsSettingsObject.h"
#include "deepdrive_simulation/PyMultiAgentSnapshotObject.h"

#include "socket/IP4ClientSocket.hpp"
#include "common/ClientErrorCode.hpp"

#include <iostream>

DeepDriveSimulation *DeepDriveSimulation::theInstance = 0;

void DeepDriveSimulation::create(const IP4Address &ip4Address)
{
	theInstance = new DeepDriveSimulation(ip4Address);
}

void DeepDriveSimulation::destroy()
{
	delete theInstance;
	theInstance = 0;
}

DeepDriveSimulation::DeepDriveSimulation(const IP4Address &ip4Address)
	: m_Socket()
{
	m_Socket.connect(ip4Address);
}

DeepDriveSimulation::~DeepDriveSimulation()
{
	m_Socket.close();
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

int32 DeepDriveSimulation::resetSimulation(float timeDilation, float startLocation)
{
	deepdrive::server::ResetSimulationRequest req;

	SimulationConfiguration &cfg = req.configuration;
	cfg.seed = 0;
	cfg.time_dilation = timeDilation;
	cfg.agent_start_location = startLocation;

	int32 res = m_Socket.send(&req, sizeof(req));
	if(res >= 0)
	{
		std::cout << "ResetSimulationRequest time dilation " << cfg.time_dilation << " start location " << cfg.agent_start_location << " sent\n";

		deepdrive::server::ResetSimulationResponse response;
		if(m_Socket.receive(&response, sizeof(response), 1000))
		{
			res = static_cast<int32> (response.result);
			std::cout << "ResetSimulationResponse received\n";
		}
		else
		{
			std::cout << "Waiting for ResetSimulationResponse, time out\n";
			res = TIME_OUT;
		}
	}

	return res;
}

int32 DeepDriveSimulation::setGraphicsSettings(PySimulationGraphicsSettingsObject *graphicsSettings)
{
	deepdrive::server::SetGraphicsSettingsRequest req;

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
		std::cout << "SetGraphicsSettingsRequest sent\n";

		deepdrive::server::SetGraphicsSettingsResponse response;
		if(m_Socket.receive(&response, sizeof(response), 1000))
		{
			res = static_cast<int32> (response.result);
			std::cout << "SetGraphicsSettingsResponse received \n";
		}
		else
		{
			res = ClientErrorCode::TIME_OUT;
			std::cout << "Waiting for SetGraphicsSettingsResponse, time out\n";
		}
	}

	return res;
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

int32 DeepDriveSimulation::getAgentsList(std::vector<uint32> &list)
{
	deepdrive::server::GetAgentsListRequest req;
	int32 res = m_Socket.send(&req, sizeof(req));
	if(res >= 0)
	{
		deepdrive::server::GetAgentsListResponse response;
		res = m_Socket.read(&response, sizeof(response), 1000);
		if (res == sizeof(response))
		{
			if (response.agent_count)
			{
				list.push_back(response.agent_ids[0]);
				for (uint32 i = 1; i < response.agent_count; ++i)
				{
					uint32 curId;
					if(m_Socket.read(&curId, sizeof(curId), 0) == sizeof(uint32))
						list.push_back(curId);
					else
						break;
				}
			}

			std::cout << "GetAgentsListResponse received containing" << list.size() << " agents\n";
			res = NO_ERROR;
		}
	}

	return res;
}

int32 DeepDriveSimulation::requestControl(const std::vector<uint32> &ids)
{
	const size_t msgSize = sizeof(deepdrive::server::RequestControlRequest) + (ids.size() > 0 ? sizeof(uint32) * (ids.size() - 1) : 0);
	int8 *msgBuf = new int8[msgSize];
	deepdrive::server::RequestControlRequest *req = new (msgBuf) deepdrive::server::RequestControlRequest(ids.size(), ids.data());

	int32 res = m_Socket.send(req, msgSize);
	if (res >= 0)
	{
		std::cout << "RequestControlRequest sent with " << req->agent_count << " agents, msgSize " << msgSize << "\n";

		deepdrive::server::RequestControlResponse response;
		if (m_Socket.receive(&response, sizeof(response), 1000))
		{
			res = static_cast<int32>(response.result);
			std::cout << "RequestControlResponse received\n";
		}
		else
		{
			std::cout << "Waiting for RequestControlResponse, time out\n";
			res = TIME_OUT;
		}
	}

	delete [] msgBuf;

	return res;
}

int32 DeepDriveSimulation::releaseControl(const std::vector<uint32> &ids)
{
	const size_t msgSize = sizeof(deepdrive::server::ReleaseControlRequest) + (ids.size() > 0 ? sizeof(uint32) * (ids.size() - 1) : 0);
	int8 *msgBuf = new int8[msgSize];
	deepdrive::server::ReleaseControlRequest *req = new (msgBuf) deepdrive::server::ReleaseControlRequest(ids.size(), ids.data());

	int32 res = m_Socket.send(req, msgSize);
	if (res >= 0)
	{
		std::cout << "ReleaseControlRequest sent with " << req->agent_count << " agents, msgSize " << msgSize << "\n";

		deepdrive::server::RequestControlResponse response;
		if (m_Socket.receive(&response, sizeof(response), 1000))
		{
			res = static_cast<int32>(response.result);
			std::cout << "ReleaseControlResponse received\n";
		}
		else
		{
			std::cout << "Waiting for ReleaseControlResponse, time out\n";
			res = TIME_OUT;
		}
	}

	delete[] msgBuf;

	return res;
}

int32 DeepDriveSimulation::setControlValues(const std::vector<deepdrive::server::SetControlValuesRequest::ControlValueSet> &controlValues)
{
	const size_t msgSize = deepdrive::server::SetControlValuesRequest::getMessageSize(controlValues.size());
	int8 *msgBuf = new int8[msgSize];
	deepdrive::server::SetControlValuesRequest *req = new (msgBuf) deepdrive::server::SetControlValuesRequest(controlValues.size(), controlValues.data());

	int32 res = m_Socket.send(req, msgSize);
	if (res >= 0)
	{
		std::cout << "SetControlValuesRequest sent with " << req->agent_count << " agents, msgSize " << msgSize << "\n";

		deepdrive::server::GenericBooleanResponse response;
		if (m_Socket.receive(&response, sizeof(response), 1000))
		{
			res = static_cast<int32>(response.result);
			std::cout << "SetControlValues response received\n";
		}
		else
		{
			std::cout << "Waiting for SetControlValues response, time out\n";
			res = TIME_OUT;
		}
	}

	delete[] msgBuf;

	return res;
}

int32 DeepDriveSimulation::step(std::vector<PyMultiAgentSnapshotObject*> &snapshots)
{
	deepdrive::server::StepRequest req;
	int32 res = m_Socket.send(&req, sizeof(req));
	if (res >= 0)
	{
		deepdrive::server::StepResponse response;
		res = m_Socket.read(&response, sizeof(response), 1000);
		if (res == sizeof(response))
		{
			if (response.agent_count)
			{
				PyMultiAgentSnapshotObject *snapshot;
				snapshot = reinterpret_cast<PyMultiAgentSnapshotObject *>(PyMultiAgentSnapshotType.tp_new(&PyMultiAgentSnapshotType, 0, 0));
				snapshots.push_back(snapshot);
				for (uint32 i = 1; i < response.agent_count; ++i)
				{
					deepdrive::server::StepResponse::SnapshotData curSnapshot;
					if (m_Socket.read(&curSnapshot, sizeof(curSnapshot), 0) == sizeof(curSnapshot))
					{
						snapshot = reinterpret_cast<PyMultiAgentSnapshotObject *>(PyMultiAgentSnapshotType.tp_new(&PyMultiAgentSnapshotType, 0, 0));
						snapshots.push_back(snapshot);
					}
					else
						break;
				}
			}

			std::cout << "Steo received containing" << snapshots.size() << " snapshots\n";
			res = NO_ERROR;
		}
	}

	return res;
}
