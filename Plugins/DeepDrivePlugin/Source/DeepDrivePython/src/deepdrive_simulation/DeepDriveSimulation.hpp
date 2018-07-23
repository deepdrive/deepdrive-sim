
#pragma once

#include "Engine.h"

#include "socket/IP4Address.hpp"
#include "socket/IP4ClientSocket.hpp"

#include "Public/Server/Messages/DeepDriveServerConfigurationMessages.h"

class DeepDriveClient;
struct PySimulationGraphicsSettingsObject;

class DeepDriveSimulation
{

public:

	DeepDriveSimulation();

	~DeepDriveSimulation();

	static int32 resetSimulation(DeepDriveClient &client, float timeDilation, float startLocation, PySimulationGraphicsSettingsObject *graphicsSettings);

	static int32 setSunSimulation(DeepDriveClient &client, uint32 month, uint32 day, uint32 minute, uint32 hour, uint32 speed);

private:

};
