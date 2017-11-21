
#pragma once

#include "Engine.h"
#include "Public/DeepDriveDataTypes.h"

enum class DeepDriveMessageType	:	uint32
{
	Undefined,
	Capture,
	Control,
	DisconnectControl,
	ConfigureCameras
};

struct DeepDriveMessageHeader
{
	DeepDriveMessageHeader(DeepDriveMessageType type, uint32 size)
		:	message_type(type)
		,	message_size(size)
		,	message_id(0)
		,	header_padding(0xEFBEADDE)
	{	}

	void setMessageId()
	{
		static uint32 nextMsgId = 1;
		message_id = nextMsgId++;
	}

	DeepDriveMessageHeader* clone() const
	{
		const uint32 msgSize = message_size + sizeof(DeepDriveMessageHeader);
		DeepDriveMessageHeader *clonedMsg = reinterpret_cast<DeepDriveMessageHeader*> (FMemory::Malloc(msgSize));

		if(clonedMsg)
			FMemory::Memcpy(clonedMsg, this, msgSize);

		return clonedMsg;
	}

	DeepDriveMessageType		message_type;
	uint32						message_size;
	uint32						message_id;
	uint32						header_padding;

};

