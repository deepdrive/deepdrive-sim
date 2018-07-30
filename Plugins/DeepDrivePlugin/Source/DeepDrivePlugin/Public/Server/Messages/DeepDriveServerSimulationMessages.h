
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


struct SetDateAndTimeRequest : public MessageHeader
{
	SetDateAndTimeRequest(uint32 _year, uint32 _month, uint32 _day, uint32 _hour, uint32 _minute)
		: MessageHeader(MessageId::SetDateAndTimeRequest, sizeof(SetDateAndTimeRequest))
		, year(_year)
		, month(_month)
		, day(_day)
		, hour(_hour)
		, minute(_minute)
	{
	}

	uint32			year;
	uint32			month;
	uint32			day;
	uint32			hour;
	uint32			minute;
};

struct SetDateAndTimeResponse : public MessageHeader
{
	SetDateAndTimeResponse(bool _result = false)
		: MessageHeader(MessageId::SetDateAndTimeResponse, sizeof(SetDateAndTimeResponse))
		, result(_result ? 1 : 0)
	{	}

	uint32		result;
};



} }	// namespaces
