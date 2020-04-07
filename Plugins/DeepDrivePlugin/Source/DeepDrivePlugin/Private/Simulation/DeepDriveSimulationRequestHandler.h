
#pragma once

#include "Engine.h"

#include "Public/Server/Messages/DeepDriveMessageIds.h"

#include <functional>
#include <map>

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveSimulationRequestHandler, Log, All);

namespace deepdrive { namespace server {
struct MessageHeader;
} }


class ADeepDriveSimulation;
class DeepDriveSimulationServer;

class DeepDriveSimulationRequestHandler
{
	typedef TQueue<deepdrive::server::MessageHeader*> MessageQueue;

	typedef std::function<void(ADeepDriveSimulation &, DeepDriveSimulationServer&, const deepdrive::server::MessageHeader &)> HandleMessageFuncPtr;
	typedef std::map<deepdrive::server::MessageId, HandleMessageFuncPtr>	MessageHandlers;

public:

	DeepDriveSimulationRequestHandler(ADeepDriveSimulation &sim, DeepDriveSimulationServer &simServer);
	~DeepDriveSimulationRequestHandler();

	void registerHandler(deepdrive::server::MessageId id, HandleMessageFuncPtr handler);

	void handleRequests();
	void enqueueRequest(deepdrive::server::MessageHeader &message);

private:

	ADeepDriveSimulation 			&m_Simulation;
	DeepDriveSimulationServer		&m_SimulationServer;

	MessageQueue					m_MessageQueue;
	MessageHandlers					m_MessageHandlers;

};
