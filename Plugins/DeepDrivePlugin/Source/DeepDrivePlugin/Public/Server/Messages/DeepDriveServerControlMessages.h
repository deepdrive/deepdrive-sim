
#pragma once

#include "Public/Server/Messages/DeepDriveServerMessageHeader.h"


namespace deepdrive { namespace server {


struct RequestAgentControlRequest :	public MessageHeader
{
	RequestAgentControlRequest(uint32 clientId)
		:	MessageHeader(MessageId::RequestAgentControlRequest, sizeof(RequestAgentControlRequest))
		,	client_id(clientId)
	{	}

	uint32		client_id;
};

struct RequestAgentControlResponse :	public MessageHeader
{
	RequestAgentControlResponse(bool granted = false)
		:	MessageHeader(MessageId::RequestAgentControlResponse, sizeof(RequestAgentControlResponse))
		,	control_granted(granted ? 1 : 0)
	{	}

	uint32		control_granted;

};

struct ReleaseAgentControlRequest :	public MessageHeader
{
	ReleaseAgentControlRequest(uint32 clientId)
		:	MessageHeader(MessageId::ReleaseAgentControlRequest, sizeof(ReleaseAgentControlRequest))
		,	client_id(clientId)
	{	}

	uint32		client_id;
};

struct ReleaseAgentControlResponse :	public MessageHeader
{
	ReleaseAgentControlResponse(bool released = false)
		:	MessageHeader(MessageId::ReleaseAgentControlResponse, sizeof(ReleaseAgentControlResponse))
		,	control_released(released ? 1 : 0)
	{	}

	uint32		control_released;

};


struct SetAgentControlValuesRequest :	public MessageHeader
{
	SetAgentControlValuesRequest(uint32 c = 0, float s = 0.0f, float t = 0.0f, float b = 0.0f, uint32 h = 0)
		:	MessageHeader(MessageId::SetAgentControlValuesRequest, sizeof(SetAgentControlValuesRequest))
		,	client_id(c)
		,	steering(s)
		,	throttle(t)
		,	brake(b)
		,	handbrake(h)
	{	}

	uint32		client_id;
	float		steering;
	float		throttle;
	float		brake;
	uint32		handbrake;

};



} }	// namespaces
