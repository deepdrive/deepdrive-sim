
#pragma once

#include "Public/Server/Messages/DeepDriveServerMessageHeader.h"
#include "Public/Simulation/DeepDriveSimulationTypes.h"

namespace deepdrive { namespace server {

struct ConfigureSimulationRequest : public MessageHeader
{
	ConfigureSimulationRequest()
		: MessageHeader(MessageId::ConfigureSimulationRequest, sizeof(ConfigureSimulationRequest))
	{	}

	SimulationConfiguration		configuration;

	SimulationGraphicsSettings	graphics_settings;
};

struct ConfigureSimulationResponse : public MessageHeader
{
	ConfigureSimulationResponse(bool _success = false)
		: MessageHeader(MessageId::ConfigureSimulationResponse, sizeof(ConfigureSimulationResponse))
		, success(_success ? 1 : 0)
	{	}

	uint32		success;
};

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
