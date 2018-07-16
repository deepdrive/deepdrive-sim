
#pragma once

#include "Engine.h"

class DeepDriveClient;

void addClient(uint32 clientId, DeepDriveClient *client);
DeepDriveClient* getClient(uint32 clientId);
bool removeClient(uint32 clientId);
