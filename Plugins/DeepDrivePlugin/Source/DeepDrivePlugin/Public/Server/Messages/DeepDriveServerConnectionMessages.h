
#pragma once

#include "Public/Server/Messages/DeepDriveServerMessageHeader.h"


namespace deepdrive { namespace server {


struct RegisterClientRequest	:	public MessageHeader
{
	RegisterClientRequest(bool master)
		:	MessageHeader(MessageId::RegisterClientRequest, sizeof(RegisterClientRequest))
		,	client_protocol_version(1)
		,	request_master_role(master ? 1 : 0)
	{	}

	uint32				client_protocol_version;
	uint32				request_master_role;

};

struct RegisterClientResponse	:	public MessageHeader
{
	enum
	{
		SharedMemNameSize = 128
	};

	RegisterClientResponse()
		:	MessageHeader(MessageId::RegisterClientResponse, sizeof(RegisterClientResponse))
	{	}

	uint32				client_id;
	uint32				granted_master_role;

	uint32				server_protocol_version;

	char				shared_memory_name[SharedMemNameSize];
	uint32				shared_memory_size;

	uint16				max_supported_cameras;
	uint16				max_capture_resolution;

	uint32				inactivity_timeout_ms;

};


struct KeepAliveRequest	:	public MessageHeader
{
	KeepAliveRequest()
		:	MessageHeader(MessageId::KeepAliveRequest, sizeof(KeepAliveRequest))
	{	}

	uint32		client_id;
};

struct KeepAliveResponse	:	public MessageHeader
{
	KeepAliveResponse()
		:	MessageHeader(MessageId::KeepAliveResponse, sizeof(KeepAliveResponse))
	{	}

	uint32		acknowledged;
};

struct UnregisterClientRequest : public MessageHeader
{
	UnregisterClientRequest(uint32 clientId)
		:	MessageHeader(MessageId::UnregisterClientRequest, sizeof(UnregisterClientRequest))
		,	client_id(clientId)
	{	}

	uint32		client_id;

};

struct UnregisterClientResponse : public MessageHeader
{
	UnregisterClientResponse()
		: MessageHeader(MessageId::UnregisterClientResponse, sizeof(UnregisterClientResponse))
	{	}
};




} }	// namespaces
