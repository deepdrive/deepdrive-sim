
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
