
#pragma once

#include "Engine.h"
#include "Public/Server/Messages/DeepDriveMessageIds.h"

namespace deepdrive { namespace server {

struct MessageHeader
{
	MessageHeader(MessageId id, uint32 size)
		:	message_id(id)
		,	message_size(size)
	{	}

	MessageHeader* clone() const
	{
		MessageHeader *clonedMsg = reinterpret_cast<MessageHeader*> (FMemory::Malloc(message_size));

		if(clonedMsg)
			FMemory::Memcpy(clonedMsg, this, message_size);

		return clonedMsg;
	}


	MessageId					message_id;
	uint32						message_size;
};

} }	// namespaces
