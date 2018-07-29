
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

	int32 configureSimulation(uint32 seed, float timeDilation, float startLocation, PySimulationGraphicsSettingsObject *graphicsSettings);

	int32 resetSimulation(float timeDilation, float startLocation, PySimulationGraphicsSettingsObject *graphicsSettings);

	int32 setSunSimulation(uint32 month, uint32 day, uint32 minute, uint32 hour, uint32 speed);

	bool isConnected() const;

private:

	IP4ClientSocket					m_Socket;

};


inline bool DeepDriveSimulation::isConnected() const
{
	return m_Socket.isConnected();
}
