
#pragma once

#include "Public/Server/Messages/DeepDriveServerMessageHeader.h"

#include <cstring>

namespace deepdrive { namespace server {

struct InitializeSimulationRequest : public MessageHeader
{
	InitializeSimulationRequest(uint32 clientId, uint32 _seed, float timeDilation, float agentStartLocation)
		: MessageHeader(MessageId::InitializeSimulationRequest, sizeof(InitializeSimulationRequest))
		, client_id(clientId)
		, seed(_seed)
		, time_dilation(timeDilation)
		, agent_start_location(agentStartLocation)
	{
	}

	uint32			client_id;
	uint32			seed;
	float			time_dilation;
	float			agent_start_location;
};

struct InitializeSimulationResponse : public MessageHeader
{
	InitializeSimulationResponse(bool _initialized = false)
		: MessageHeader(MessageId::InitializeSimulationResponse, sizeof(InitializeSimulationResponse))
		, initialized(_initialized ? 1 : 0)
	{	}

	uint32		initialized;

};

struct SetSunSimulationRequest : public MessageHeader
{
	SetSunSimulationRequest(uint32 clientId, uint32 _month, uint32 _day, uint32 _hour, uint32 _minute, uint32 _speed)
		: MessageHeader(MessageId::SetSunSimulationRequest, sizeof(SetSunSimulationRequest))
		, client_id(clientId)
		, month(_month)
		, day(_day)
		, hour(_hour)
		, minute(_minute)
		, speed(_speed)
	{
	}

	uint32			client_id;
	uint32			month;
	uint32			day;
	uint32			hour;
	uint32			minute;
	uint32			speed;
};

struct SetSunSimulationResponse : public MessageHeader
{
	SetSunSimulationResponse(bool _result = false)
		: MessageHeader(MessageId::SetSunSimulationResponse, sizeof(SetSunSimulationResponse))
		, result(_result ? 1 : 0)
	{	}

	uint32		result;
};

struct RegisterCaptureCameraRequest	:	public MessageHeader
{
	RegisterCaptureCameraRequest(uint32 clientId, float hFoV, uint16 captureWidth, uint16 captureHeight, const char *label)
		:	MessageHeader(MessageId::RegisterCaptureCameraRequest, sizeof(RegisterCaptureCameraRequest))
		,	client_id(clientId)
		,	horizontal_field_of_view(hFoV)
		,	capture_width(captureWidth)
		,	capture_height(captureHeight)
	{
		if(label)
		{
			strncpy(camera_label, label, MessageHeader::StringSize - 1);
			camera_label[MessageHeader::StringSize - 1] = 0;
		}
		else
			camera_label[0] = 0;
	}

	uint32				client_id;
	float				horizontal_field_of_view;
	uint16				capture_width;
	uint16				capture_height;
 	float				relative_position[3];
	float				relative_rotation[3];

	char				camera_label[MessageHeader::StringSize];
};

struct RegisterCaptureCameraResponse	:	public MessageHeader
{
	RegisterCaptureCameraResponse(uint32 camId = 0)
		:	MessageHeader(MessageId::RegisterCaptureCameraResponse, sizeof(RegisterCaptureCameraResponse))
		,	camera_id(camId)
	{	}

	uint32		camera_id;

};



} }	// namespaces
