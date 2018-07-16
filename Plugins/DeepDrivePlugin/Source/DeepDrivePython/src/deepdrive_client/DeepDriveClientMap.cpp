
#include "deepdrive_client/DeepDriveClientMap.hpp"
#include "deepdrive_client/DeepDriveClient.hpp"

#include <map>

typedef std::map<uint32, DeepDriveClient*>		ClientMap;

static ClientMap g_Clients;


void addClient(uint32 clientId, DeepDriveClient *client)
{
	g_Clients[clientId] = client;
}

DeepDriveClient* getClient(uint32 clientId)
{
	ClientMap::iterator cIt = g_Clients.find(clientId);
	DeepDriveClient *client = cIt != g_Clients.end() ? cIt->second : 0;
	return client;
}

bool removeClient(uint32 clientId)
{
	bool removed = false;
	ClientMap::iterator cIt = g_Clients.find(clientId);
	if(cIt != g_Clients.end())
	{
		DeepDriveClient *client = cIt->second;
		if(client)
		{
			client->close();
			delete client;
		}
		g_Clients.erase(cIt);
		removed = true;
	}
	return removed;
}
