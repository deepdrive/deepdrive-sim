
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
	ResetSimulationRequest()
		: MessageHeader(MessageId::ResetSimulationRequest, sizeof(ResetSimulationRequest))
	{	}

	SimulationConfiguration		configuration;
};

struct ResetSimulationResponse : public MessageHeader
{
	ResetSimulationResponse(bool _result = false)
		: MessageHeader(MessageId::ResetSimulationResponse, sizeof(ResetSimulationResponse))
		, result(_result ? 1 : 0)
	{	}

	uint32		result;
};

struct SetGraphicsSettingsRequest : public MessageHeader
{
	SetGraphicsSettingsRequest()
		: MessageHeader(MessageId::SetGraphicsSettingsRequest, sizeof(SetGraphicsSettingsRequest))
	{	}

	SimulationGraphicsSettings	graphics_settings;
};

struct SetGraphicsSettingsResponse : public MessageHeader
{
	SetGraphicsSettingsResponse(bool _result = false)
		: MessageHeader(MessageId::SetGraphicsSettingsResponse, sizeof(SetGraphicsSettingsResponse))
		, result(_result ? 1 : 0)
	{	}

	uint32		result;
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


struct SetSunSimulationSpeedRequest : public MessageHeader
{
	SetSunSimulationSpeedRequest(uint32 _speed)
		: MessageHeader(MessageId::SetSunSimulationSpeedRequest, sizeof(SetSunSimulationSpeedRequest))
		, speed(_speed)
	{
	}

	uint32			speed;
};


struct SetSunSimulationSpeedResponse : public MessageHeader
{
	SetSunSimulationSpeedResponse(bool _result = false)
		: MessageHeader(MessageId::SetSunSimulationSpeedResponse, sizeof(SetSunSimulationSpeedResponse))
		, result(_result ? 1 : 0)
	{	}

	uint32		result;
};

} }	// namespaces
