
#pragma once

#include "Engine.h"

#include "socket/IP4Address.hpp"
#include "socket/IP4ClientSocket.hpp"

#include "Public/Server/Messages/DeepDriveServerConfigurationMessages.h"

struct PySimulationGraphicsSettingsObject;

class DeepDriveSimulation
{

public:

	DeepDriveSimulation(const IP4Address &ip4Address);

	~DeepDriveSimulation();

	static int32 resetSimulation(float timeDilation, float startLocation, PySimulationGraphicsSettingsObject *graphicsSettings);

	static int32 setSunSimulation(uint32 month, uint32 day, uint32 minute, uint32 hour, uint32 speed);

private:

	IP4ClientSocket					m_Socket;

};
