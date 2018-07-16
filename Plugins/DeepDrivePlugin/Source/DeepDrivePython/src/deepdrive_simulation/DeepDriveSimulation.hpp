
#pragma once

#include "Engine.h"

#include "socket/IP4Address.hpp"
#include "socket/IP4ClientSocket.hpp"

#include "Public/Server/Messages/DeepDriveServerConfigurationMessages.h"

class DeepDriveClient;

class DeepDriveSimulation
{

public:

	DeepDriveSimulation();

	~DeepDriveSimulation();

	static int32 initialize(DeepDriveClient &client, uint32 seed, float timeDilation, float startLocation);

	static int32 setSunSimulation(DeepDriveClient &client, uint32 month, uint32 day, uint32 minute, uint32 hour, uint32 speed);

private:

};
