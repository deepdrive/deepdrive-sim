
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


struct ResetAgentRequest : public MessageHeader
{
	ResetAgentRequest(uint32 clientId)
		: MessageHeader(MessageId::ResetAgentRequest, sizeof(ResetAgentRequest))
		, client_id(clientId)
	{	}

	uint32		client_id;
};

struct ResetAgentResponse : public MessageHeader
{
	ResetAgentResponse(bool _reset = false)
		: MessageHeader(MessageId::ResetAgentResponse, sizeof(ResetAgentResponse))
		, reset(_reset ? 1 : 0)
	{	}

	uint32		reset;
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


struct ActivateSynchronousSteppingRequest :	public MessageHeader
{
	ActivateSynchronousSteppingRequest(uint32 clientId)
		:	MessageHeader(MessageId::ActivateSynchronousSteppingRequest, sizeof(ActivateSynchronousSteppingRequest))
		,	client_id(clientId)
	{	}

	uint32		client_id;
};

struct ActivateSynchronousSteppingResponse :	public MessageHeader
{
	ActivateSynchronousSteppingResponse(bool activated = false)
		:	MessageHeader(MessageId::ActivateSynchronousSteppingResponse, sizeof(ActivateSynchronousSteppingResponse))
		,	synchronous_stepping_activated(activated ? 1 : 0)
	{	}

	uint32		synchronous_stepping_activated;

};


struct DeactivateSynchronousSteppingRequest :	public MessageHeader
{
	DeactivateSynchronousSteppingRequest(uint32 clientId)
		:	MessageHeader(MessageId::DeactivateSynchronousSteppingRequest, sizeof(DeactivateSynchronousSteppingRequest))
		,	client_id(clientId)
	{	}

	uint32		client_id;
};

struct DeactivateSynchronousSteppingResponse :	public MessageHeader
{
	DeactivateSynchronousSteppingResponse(bool deactivated = false)
		:	MessageHeader(MessageId::DeactivateSynchronousSteppingResponse, sizeof(DeactivateSynchronousSteppingResponse))
		,	synchronous_stepping_deactivated(deactivated ? 1 : 0)
	{	}

	uint32		synchronous_stepping_deactivated;

};


struct AdvanceSynchronousSteppingRequest :	public MessageHeader
{
	AdvanceSynchronousSteppingRequest(uint32 c = 0, float dt = 0.0f, float s = 0.0f, float t = 0.0f, float b = 0.0f, uint32 h = 0)
		:	MessageHeader(MessageId::AdvanceSynchronousSteppingRequest, sizeof(AdvanceSynchronousSteppingRequest))
		,	client_id(c)
		,	time_step(dt)
		,	steering(s)
		,	throttle(t)
		,	brake(b)
		,	handbrake(h)
	{	}

	uint32		client_id;
	float		time_step;
	float		steering;
	float		throttle;
	float		brake;
	uint32		handbrake;

};


struct AdvanceSynchronousSteppingResponse :	public MessageHeader
{
	AdvanceSynchronousSteppingResponse(int32 seqNr = 0)
		:	MessageHeader(MessageId::AdvanceSynchronousSteppingResponse, sizeof(AdvanceSynchronousSteppingResponse))
		,	sequence_number(seqNr)
	{	}

	int32		sequence_number;

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
