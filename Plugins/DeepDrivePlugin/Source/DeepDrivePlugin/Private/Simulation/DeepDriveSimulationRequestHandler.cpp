
#include "DeepDrivePluginPrivatePCH.h"
#include "Private/Simulation/DeepDriveSimulationRequestHandler.h"
#include "Private/Server/DeepDriveSimulationServer.h"
#include "Public/Server/Messages/DeepDriveServerSimulationMessages.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"
#include "Public/Simulation/Agent/DeepDriveAgentControllerBase.h"
#include "Public/Simulation/Misc/DeepDriveRandomStream.h"

#include "Public/Simulation/DeepDriveSimulation.h"

DEFINE_LOG_CATEGORY(LogDeepDriveSimulationRequestHandler);

DeepDriveSimulationRequestHandler::DeepDriveSimulationRequestHandler(ADeepDriveSimulation &sim, DeepDriveSimulationServer &simServer)
	:	m_Simulation(sim)
	,	m_SimulationServer(simServer)
{

}

DeepDriveSimulationRequestHandler::~DeepDriveSimulationRequestHandler()
{
}

void DeepDriveSimulationRequestHandler::registerHandler(deepdrive::server::MessageId id, HandleMessageFuncPtr handler)
{
	m_MessageHandlers[id] = handler;
}

void DeepDriveSimulationRequestHandler::handleRequests()
{
	deepdrive::server::MessageHeader *message = 0;
	if (	m_MessageQueue.Dequeue(message)
		&&	message
		)
	{
		MessageHandlers::iterator fIt = m_MessageHandlers.find(message->message_id);
		if (fIt != m_MessageHandlers.end())
			fIt->second(m_Simulation, m_SimulationServer, * message);

		FMemory::Free(message);
	}
}

void DeepDriveSimulationRequestHandler::enqueueRequest(deepdrive::server::MessageHeader &message)
{
	m_MessageQueue.Enqueue(&message);
}
