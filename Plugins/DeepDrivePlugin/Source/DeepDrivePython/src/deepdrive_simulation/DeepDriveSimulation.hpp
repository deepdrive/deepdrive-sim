
#pragma once

#include "Engine.h"

#include "socket/IP4Address.hpp"
#include "socket/IP4ClientSocket.hpp"

#include "Public/Server/Messages/DeepDriveServerConfigurationMessages.h"
#include "Public/Server/Messages/DeepDriveServerSimulationMessages.h"

#include <vector>

struct PySimulationGraphicsSettingsObject;
struct PyMultiAgentSnapshotObject;

class DeepDriveSimulation
{

public:

	static DeepDriveSimulation* getInstance()
	{
		return theInstance;
	}

	static void create(const IP4Address &ip4Address);

	static void destroy();

	int32 configureSimulation(uint32 seed, float timeDilation, float startLocation, PySimulationGraphicsSettingsObject *graphicsSettings);

	int32 resetSimulation(float timeDilation, float startLocation);

	int32 setGraphicsSettings(PySimulationGraphicsSettingsObject *graphicsSettings);

	int32 setDateAndTime(uint32 year, uint32 month, uint32 day, uint32 minute, uint32 hour);
	
	int32 setSpeed(uint32 speed);

	int32 getAgentsList(std::vector<uint32> &list);

	int32 requestControl(const std::vector<uint32> &ids);

	int32 releaseControl(const std::vector<uint32> &ids);

	int32 setControlValues(const std::vector<deepdrive::server::SetControlValuesRequest::ControlValueSet> &controlValues);

	int32 step(std::vector<PyMultiAgentSnapshotObject*> &snapshots);

	bool isConnected() const;

private:

	DeepDriveSimulation(const IP4Address &ip4Address);

	~DeepDriveSimulation();

	static DeepDriveSimulation		*theInstance;

	IP4ClientSocket					m_Socket;

};


inline bool DeepDriveSimulation::isConnected() const
{
	return m_Socket.isConnected();
}
