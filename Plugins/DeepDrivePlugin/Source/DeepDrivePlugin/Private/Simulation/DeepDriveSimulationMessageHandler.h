
#pragma once

#include "Engine.h"

#include "Public/Server/Messages/DeepDriveMessageIds.h"

#include <functional>
#include <map>

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveSimulationMessageHandler, Log, All);

namespace deepdrive { namespace server {
struct MessageHeader;
} }


class ADeepDriveSimulation;
class DeepDriveSimulationServer;

class DeepDriveSimulationMessageHandler
{
	typedef TQueue<deepdrive::server::MessageHeader*> MessageQueue;

	typedef std::function< void(const deepdrive::server::MessageHeader&) > HandleMessageFuncPtr;
	typedef std::map<deepdrive::server::MessageId, HandleMessageFuncPtr>	MessageHandlers;

public:

	DeepDriveSimulationMessageHandler(ADeepDriveSimulation &sim, DeepDriveSimulationServer &simServer);
	~DeepDriveSimulationMessageHandler();

	void handleMessages();
	void enqueueMessage(deepdrive::server::MessageHeader &message);

private:

	void configure(const deepdrive::server::MessageHeader& message);
	void reset(const deepdrive::server::MessageHeader& message);
	void setGfxSettings(const deepdrive::server::MessageHeader& message);
	void setDateAndTime(const deepdrive::server::MessageHeader& message);
	void setSunSimulationSpeed(const deepdrive::server::MessageHeader& message);

	ADeepDriveSimulation 			&m_Simulation;
	DeepDriveSimulationServer		&m_SimulationServer;

	MessageQueue					m_MessageQueue;
	MessageHandlers					m_MessageHandlers;

};
