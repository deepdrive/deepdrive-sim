
#pragma once

#include "Engine.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveMultiAgentRequests, Log, All);

namespace deepdrive { namespace server {
struct MessageHeader;
} }


class ADeepDriveSimulation;
class DeepDriveSimulationServer;
class DeepDriveSimulationRequestHandler;

class DeepDriveMultiAgentRequests
{
public:

	static void registerHandlers(DeepDriveSimulationRequestHandler &requestHandler);

private:

	static void getAgentsList(ADeepDriveSimulation &simulation, DeepDriveSimulationServer &simulationServer, const deepdrive::server::MessageHeader &message);

	static void requestControl(ADeepDriveSimulation &simulation, DeepDriveSimulationServer &simulationServer, const deepdrive::server::MessageHeader &message);

	static void releaseControl(ADeepDriveSimulation &simulation, DeepDriveSimulationServer &simulationServer, const deepdrive::server::MessageHeader &message);

	static void setControlValues(ADeepDriveSimulation &simulation, DeepDriveSimulationServer &simulationServer, const deepdrive::server::MessageHeader &message);

	static void step(ADeepDriveSimulation &simulation, DeepDriveSimulationServer &simulationServer, const deepdrive::server::MessageHeader &message);
};
