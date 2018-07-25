
#pragma once

#include "Public/Server/Messages/DeepDriveServerMessageHeader.h"
#include "Public/Simulation/DeepDriveSimulationTypes.h"

namespace deepdrive { namespace server {

struct ResetSimulationRequest : public MessageHeader
{
	ResetSimulationRequest(uint32 clientId)
		: MessageHeader(MessageId::ResetSimulationRequest, sizeof(ResetSimulationRequest))
		, client_id(clientId)
	{	}

	uint32						client_id;

	SimulationConfiguration		configuration;

	SimulationGraphicsSettings	graphics_settings;
};

struct ResetSimulationResponse : public MessageHeader
{
	ResetSimulationResponse(bool _reset = false)
		: MessageHeader(MessageId::ResetSimulationResponse, sizeof(ResetSimulationResponse))
		, reset(_reset ? 1 : 0)
	{	}

	uint32		reset;
};


} }	// namespaces
