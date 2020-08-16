
#pragma once

#include "Engine.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveSimulationRequests, Log, All);

namespace deepdrive { namespace server {
struct MessageHeader;
} }


class ADeepDriveSimulation;
class DeepDriveSimulationServer;
class DeepDriveSimulationRequestHandler;

class DeepDriveSimulationRequests
{
public:

	static void registerHandlers(DeepDriveSimulationRequestHandler &requestHandler);

private:

	static void configure(ADeepDriveSimulation &simulation, DeepDriveSimulationServer &simulationServer, const deepdrive::server::MessageHeader &message);
	static void reset(ADeepDriveSimulation &simulation, DeepDriveSimulationServer &simulationServer, const deepdrive::server::MessageHeader& message);
	static void setGfxSettings(ADeepDriveSimulation &simulation, DeepDriveSimulationServer &simulationServer, const deepdrive::server::MessageHeader &message);
	static void setDateAndTime(ADeepDriveSimulation &simulation, DeepDriveSimulationServer &simulationServer, const deepdrive::server::MessageHeader &message);
	static void setSunSimulationSpeed(ADeepDriveSimulation &simulation, DeepDriveSimulationServer &simulationServer, const deepdrive::server::MessageHeader& message);

};
