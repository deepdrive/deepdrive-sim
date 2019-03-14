
#include "DeepDrivePluginPrivatePCH.h"
#include "Private/Simulation/DeepDriveSimulationMessageHandler.h"
#include "Private/Server/DeepDriveSimulationServer.h"
#include "Public/Server/Messages/DeepDriveServerSimulationMessages.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"
#include "Public/Simulation/Agent/DeepDriveAgentControllerBase.h"
#include "Public/Simulation/Misc/DeepDriveRandomStream.h"

#include "Public/Simulation/DeepDriveSimulation.h"

DEFINE_LOG_CATEGORY(LogDeepDriveSimulationMessageHandler);

DeepDriveSimulationMessageHandler::DeepDriveSimulationMessageHandler(ADeepDriveSimulation &sim, DeepDriveSimulationServer &simServer)
	:	m_Simulation(sim)
	,	m_SimulationServer(simServer)
{

	m_MessageHandlers[deepdrive::server::MessageId::ConfigureSimulationRequest] = std::bind(&DeepDriveSimulationMessageHandler::configure, this, std::placeholders::_1);
	m_MessageHandlers[deepdrive::server::MessageId::ResetSimulationRequest] = std::bind(&DeepDriveSimulationMessageHandler::reset, this, std::placeholders::_1);
	m_MessageHandlers[deepdrive::server::MessageId::SetGraphicsSettingsRequest] = std::bind(&DeepDriveSimulationMessageHandler::setGfxSettings, this, std::placeholders::_1);
	m_MessageHandlers[deepdrive::server::MessageId::SetDateAndTimeRequest] = std::bind(&DeepDriveSimulationMessageHandler::setDateAndTime, this, std::placeholders::_1);
	m_MessageHandlers[deepdrive::server::MessageId::SetSunSimulationSpeedRequest] = std::bind(&DeepDriveSimulationMessageHandler::setSunSimulationSpeed, this, std::placeholders::_1);

}

DeepDriveSimulationMessageHandler::~DeepDriveSimulationMessageHandler()
{
}

void DeepDriveSimulationMessageHandler::handleMessages()
{
	deepdrive::server::MessageHeader *message = 0;
	if (m_MessageQueue.Dequeue(message)
		&& message
		)
	{
		MessageHandlers::iterator fIt = m_MessageHandlers.find(message->message_id);
		if (fIt != m_MessageHandlers.end())
			fIt->second(*message);

		FMemory::Free(message);
	}
}

void DeepDriveSimulationMessageHandler::enqueueMessage(deepdrive::server::MessageHeader &message)
{
	m_MessageQueue.Enqueue(&message);
}


void DeepDriveSimulationMessageHandler::configure(const deepdrive::server::MessageHeader& message)
{
	UE_LOG(LogDeepDriveSimulationMessageHandler, Log, TEXT("DeepDriveSimulation configure") );

	const deepdrive::server::ConfigureSimulationRequest &req = static_cast<const deepdrive::server::ConfigureSimulationRequest&> (message);
	const SimulationConfiguration &configuration = req.configuration;
	const SimulationGraphicsSettings &graphicsSettings = req.graphics_settings;

	m_Simulation.Seed = configuration.seed;
	for (auto &rsd : m_Simulation.RandomStreams)
		rsd.Value.getRandomStream()->initialize(configuration.seed);

	for (auto &agent : m_Simulation.m_Agents)
	{
		agent->getAgentController()->OnConfigureSimulation(configuration, true);
	}

	m_Simulation.applyGraphicsSettings(graphicsSettings);

	m_SimulationServer.enqueueResponse( new deepdrive::server::ConfigureSimulationResponse(true) );
}

void DeepDriveSimulationMessageHandler::reset(const deepdrive::server::MessageHeader& message)
{
	const deepdrive::server::ResetSimulationRequest &req = static_cast<const deepdrive::server::ResetSimulationRequest&> (message);
	const SimulationConfiguration &configuration = req.configuration;

	UE_LOG(LogDeepDriveSimulationMessageHandler, Log, TEXT("DeepDriveSimulation reset with seed %d time dilation %f agent start location %f"), configuration.seed, configuration.time_dilation, configuration.agent_start_location );

	for (auto &rsd : m_Simulation.RandomStreams)
		if(rsd.Value.ReSeedOnReset)
			rsd.Value.getRandomStream()->initialize(m_Simulation.Seed);

	for (auto &agent : m_Simulation.m_Agents)
	{
		ADeepDriveAgentControllerBase *controller = agent->getAgentController();
		if (controller)
		{
			controller->OnConfigureSimulation(configuration, false);
			controller->ResetAgent();
		}
	}

	m_SimulationServer.enqueueResponse( new deepdrive::server::ResetSimulationResponse(true) );
}

void DeepDriveSimulationMessageHandler::setGfxSettings(const deepdrive::server::MessageHeader& message)
{
	UE_LOG(LogDeepDriveSimulationMessageHandler, Log, TEXT("DeepDriveSimulation set graphics settings") );

	const deepdrive::server::SetGraphicsSettingsRequest &req = static_cast<const deepdrive::server::SetGraphicsSettingsRequest&> (message);
	m_Simulation.applyGraphicsSettings(req.graphics_settings);
	m_SimulationServer.enqueueResponse( new deepdrive::server::SetGraphicsSettingsResponse(true) );
}

void DeepDriveSimulationMessageHandler::setDateAndTime(const deepdrive::server::MessageHeader& message)
{
	const deepdrive::server::SetDateAndTimeRequest &req = static_cast<const deepdrive::server::SetDateAndTimeRequest&> (message);

	UE_LOG(LogDeepDriveSimulationMessageHandler, Log, TEXT("DeepDriveSimulation Set Date/Time to %d/%d/%d - %d/%d"), req.year, req.month, req.day, req.hour, req.minute );

	m_Simulation.SetDateAndTime(req.year, req.month, req.day, req.hour, req.minute);

	m_SimulationServer.enqueueResponse( new deepdrive::server::SetDateAndTimeResponse(true) );
}

void DeepDriveSimulationMessageHandler::setSunSimulationSpeed(const deepdrive::server::MessageHeader& message)
{
	const deepdrive::server::SetSunSimulationSpeedRequest &req = static_cast<const deepdrive::server::SetSunSimulationSpeedRequest&> (message);

	m_Simulation.SetSunSimulationSpeed(req.speed);

	m_SimulationServer.enqueueResponse( new deepdrive::server::SetSunSimulationSpeedResponse(true) );
}
