
#include "Simulation/Requests/DeepDriveMultiAgentRequests.h"
#include "Simulation/DeepDriveSimulationRequestHandler.h"
#include "Server/DeepDriveSimulationServer.h"
#include "Server/Messages/DeepDriveServerSimulationMessages.h"
#include "Simulation/Agent/DeepDriveAgent.h"
#include "Simulation/Agent/DeepDriveAgentControllerBase.h"

#include "Simulation/DeepDriveSimulation.h"

#include "Server/Messages/DeepDriveMessageIds.h"

DEFINE_LOG_CATEGORY(LogDeepDriveMultiAgentRequests);

void DeepDriveMultiAgentRequests::registerHandlers(DeepDriveSimulationRequestHandler &requestHandler)
{
	requestHandler.registerHandler(deepdrive::server::MessageId::GetAgentsListRequest, std::bind(&DeepDriveMultiAgentRequests::getAgentsList, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
	requestHandler.registerHandler(deepdrive::server::MessageId::RequestControlRequest, std::bind(&DeepDriveMultiAgentRequests::requestControl, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
	requestHandler.registerHandler(deepdrive::server::MessageId::ReleaseControlRequest, std::bind(&DeepDriveMultiAgentRequests::releaseControl, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
	requestHandler.registerHandler(deepdrive::server::MessageId::SetControlValuesRequest, std::bind(&DeepDriveMultiAgentRequests::setControlValues, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
}

void DeepDriveMultiAgentRequests::getAgentsList(ADeepDriveSimulation &simulation, DeepDriveSimulationServer &simulationServer, const deepdrive::server::MessageHeader& message)
{
	UE_LOG(LogDeepDriveMultiAgentRequests, Log, TEXT("DeepDriveSimulation getAgentsList") );

	TArray<uint32> ids;

	for(auto agent : simulation.m_Agents)
	{
		auto id = agent->GetAgentId();
		ids.Add(agent->GetAgentId());
	}

	const size_t msgSize = deepdrive::server::GetAgentsListResponse::getMessageSize(ids.Num());
	deepdrive::server::GetAgentsListResponse *response = new (FMemory::Malloc(msgSize, 4)) deepdrive::server::GetAgentsListResponse(ids.Num(), ids.GetData());

	simulationServer.enqueueResponse(response);
}

void DeepDriveMultiAgentRequests::requestControl(ADeepDriveSimulation &simulation, DeepDriveSimulationServer &simulationServer, const deepdrive::server::MessageHeader &message)
{
	const deepdrive::server::RequestControlRequest &req = static_cast<const deepdrive::server::RequestControlRequest&> (message);

	UE_LOG(LogDeepDriveMultiAgentRequests, Log, TEXT("request control for Agent Count  %d"), req.agent_count);

	if(req.agent_count == 0)
	{
		UE_LOG(LogDeepDriveMultiAgentRequests, Log, TEXT("Granting control for all agents"));
		for (auto agent : simulation.m_Agents)
		{
			agent->getAgentController()->RequestControl();
		}
	}
	else
	{
		for (uint32 i = 0; i < req.agent_count; ++i)
		{
			const uint32 id = req.agent_ids[i];
			const int32 index = id - 1;
			if(index >= 0 && index < simulation.m_Agents.Num())
				simulation.m_Agents[index]->getAgentController()->RequestControl();
		}
	}

	simulationServer.enqueueResponse(new deepdrive::server::RequestControlResponse(true));
}

void DeepDriveMultiAgentRequests::releaseControl(ADeepDriveSimulation &simulation, DeepDriveSimulationServer &simulationServer, const deepdrive::server::MessageHeader &message)
{
	const deepdrive::server::ReleaseControlRequest &req = static_cast<const deepdrive::server::ReleaseControlRequest &>(message);

	UE_LOG(LogDeepDriveMultiAgentRequests, Log, TEXT("release control for Agent Count  %d"), req.agent_count);

	if (req.agent_count == 0)
	{
		UE_LOG(LogDeepDriveMultiAgentRequests, Log, TEXT("Releasing control for all agents"));
		for (auto agent : simulation.m_Agents)
		{
			agent->getAgentController()->ReleaseControl();
		}
	}
	else
	{
		for (uint32 i = 0; i < req.agent_count; ++i)
		{
			const uint32 id = req.agent_ids[i];
			const int32 index = id - 1;
			if (index >= 0 && index < simulation.m_Agents.Num())
				simulation.m_Agents[index]->getAgentController()->ReleaseControl();
		}
	}

	simulationServer.enqueueResponse(new deepdrive::server::ReleaseControlResponse(true));
}

void DeepDriveMultiAgentRequests::setControlValues(ADeepDriveSimulation &simulation, DeepDriveSimulationServer &simulationServer, const deepdrive::server::MessageHeader &message)
{
	const deepdrive::server::SetControlValuesRequest &req = static_cast<const deepdrive::server::SetControlValuesRequest &>(message);

	UE_LOG(LogDeepDriveMultiAgentRequests, Log, TEXT("Set control values for Agent Count  %d"), req.agent_count);

	for (uint32 i = 0; i < req.agent_count; ++i)
	{
		for (uint32 i = 0; i < req.agent_count; ++i)
		{
			const uint32 id = req.control_values[i].agent_id;
			const int32 index = id - 1;
			if	(	index >= 0
				&&	index < simulation.m_Agents.Num()
				)
			{
				ADeepDriveAgentControllerBase *controller = simulation.m_Agents[index]->getAgentController();
				if(controller && controller->isRemotelyControlled())
					controller->SetControlValues(req.control_values[i].steering, req.control_values[i].throttle, req.control_values[i].brake, req.control_values[i].handbrake != 0);
				else
					UE_LOG(LogDeepDriveMultiAgentRequests, Log, TEXT("Agent %d has no controller (%p) or isn't remotely controlled"), id, controller);
			}
			else
				UE_LOG(LogDeepDriveMultiAgentRequests, Log, TEXT("Unknown agent %d"), id);
		}
	}

	simulationServer.enqueueResponse(new deepdrive::server::GenericBooleanResponse(true));
}

void DeepDriveMultiAgentRequests::step(ADeepDriveSimulation &simulation, DeepDriveSimulationServer &simulationServer, const deepdrive::server::MessageHeader &message)
{

}
