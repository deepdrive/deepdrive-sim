
#include "DeepDrivePluginPrivatePCH.h"
#include "Private/Simulation/Requests/DeepDriveSimulationRequests.h"
#include "Private/Simulation/DeepDriveSimulationRequestHandler.h"
#include "Private/Server/DeepDriveSimulationServer.h"
#include "Public/Server/Messages/DeepDriveServerSimulationMessages.h"
#include "Public/Simulation/Agent/DeepDriveAgent.h"
#include "Public/Simulation/Agent/DeepDriveAgentControllerBase.h"
#include "Public/Simulation/Misc/DeepDriveRandomStream.h"

#include "Public/Simulation/DeepDriveSimulation.h"

#include "Public/Server/Messages/DeepDriveMessageIds.h"

DEFINE_LOG_CATEGORY(LogDeepDriveSimulationRequests);

void DeepDriveSimulationRequests::registerHandlers(DeepDriveSimulationRequestHandler &requestHandler)
{
	requestHandler.registerHandler(deepdrive::server::MessageId::ConfigureSimulationRequest, std::bind(&DeepDriveSimulationRequests::configure, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
	requestHandler.registerHandler(deepdrive::server::MessageId::ResetSimulationRequest, std::bind(&DeepDriveSimulationRequests::reset, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
	requestHandler.registerHandler(deepdrive::server::MessageId::SetGraphicsSettingsRequest, std::bind(&DeepDriveSimulationRequests::setGfxSettings, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
	requestHandler.registerHandler(deepdrive::server::MessageId::SetDateAndTimeRequest, std::bind(&DeepDriveSimulationRequests::setDateAndTime, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
	requestHandler.registerHandler(deepdrive::server::MessageId::SetSunSimulationSpeedRequest, std::bind(&DeepDriveSimulationRequests::setSunSimulationSpeed, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
}


void DeepDriveSimulationRequests::configure(ADeepDriveSimulation &simulation, DeepDriveSimulationServer &simulationServer, const deepdrive::server::MessageHeader& message)
{
	UE_LOG(LogDeepDriveSimulationRequests, Log, TEXT("DeepDriveSimulation configure") );

	const deepdrive::server::ConfigureSimulationRequest &req = static_cast<const deepdrive::server::ConfigureSimulationRequest&> (message);
	const SimulationConfiguration &configuration = req.configuration;
	const SimulationGraphicsSettings &graphicsSettings = req.graphics_settings;

	simulation.Seed = configuration.seed;
	for (auto &rsd : simulation.RandomStreams)
		rsd.Value.getRandomStream()->initialize(configuration.seed);

	for (auto &agent : simulation.m_Agents)
	{
		agent->getAgentController()->OnConfigureSimulation(configuration, true);
	}

	simulation.applyGraphicsSettings(graphicsSettings);

	simulationServer.enqueueResponse(new deepdrive::server::ConfigureSimulationResponse(true));
}

void DeepDriveSimulationRequests::reset(ADeepDriveSimulation &simulation, DeepDriveSimulationServer &simulationServer, const deepdrive::server::MessageHeader &message)
{
	const deepdrive::server::ResetSimulationRequest &req = static_cast<const deepdrive::server::ResetSimulationRequest&> (message);
	const SimulationConfiguration &configuration = req.configuration;

	UE_LOG(LogDeepDriveSimulationRequests, Log, TEXT("DeepDriveSimulation reset with seed %d time dilation %f agent start location %f"), configuration.seed, configuration.time_dilation, configuration.agent_start_location );

	for (auto &rsd : simulation.RandomStreams)
		if(rsd.Value.ReSeedOnReset)
			rsd.Value.getRandomStream()->initialize(simulation.Seed);

	for (auto &agent : simulation.m_Agents)
	{
		ADeepDriveAgentControllerBase *controller = agent->getAgentController();
		if (controller)
		{
			controller->OnConfigureSimulation(configuration, false);
			controller->Reset();
		}
	}

	simulationServer.enqueueResponse( new deepdrive::server::ResetSimulationResponse(true) );
}

void DeepDriveSimulationRequests::setGfxSettings(ADeepDriveSimulation &simulation, DeepDriveSimulationServer &simulationServer, const deepdrive::server::MessageHeader &message)
{
	UE_LOG(LogDeepDriveSimulationRequests, Log, TEXT("DeepDriveSimulation set graphics settings") );

	const deepdrive::server::SetGraphicsSettingsRequest &req = static_cast<const deepdrive::server::SetGraphicsSettingsRequest&> (message);
	simulation.applyGraphicsSettings(req.graphics_settings);
	simulationServer.enqueueResponse( new deepdrive::server::SetGraphicsSettingsResponse(true) );
}

void DeepDriveSimulationRequests::setDateAndTime(ADeepDriveSimulation &simulation, DeepDriveSimulationServer &simulationServer, const deepdrive::server::MessageHeader &message)
{
	const deepdrive::server::SetDateAndTimeRequest &req = static_cast<const deepdrive::server::SetDateAndTimeRequest&> (message);

	UE_LOG(LogDeepDriveSimulationRequests, Log, TEXT("DeepDriveSimulation Set Date/Time to %d/%d/%d - %d/%d"), req.year, req.month, req.day, req.hour, req.minute );

	simulation.SetDateAndTime(req.year, req.month, req.day, req.hour, req.minute);

	simulationServer.enqueueResponse( new deepdrive::server::SetDateAndTimeResponse(true) );
}

void DeepDriveSimulationRequests::setSunSimulationSpeed(ADeepDriveSimulation &simulation, DeepDriveSimulationServer &simulationServer, const deepdrive::server::MessageHeader &message)
{
	const deepdrive::server::SetSunSimulationSpeedRequest &req = static_cast<const deepdrive::server::SetSunSimulationSpeedRequest&> (message);

	simulation.SetSunSimulationSpeed(req.speed);

	simulationServer.enqueueResponse( new deepdrive::server::SetSunSimulationSpeedResponse(true) );
}

