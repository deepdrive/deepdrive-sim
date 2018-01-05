
#pragma once

#include "Engine.h"

struct IP4Address;

class IIP4ClientSocketImpl
{
public:

	virtual bool connect(const IP4Address &ip4Address) = 0;

	virtual uint32 send(const void *data, uint32 bytesToSend) = 0;

	virtual uint32 receive(void *buffer, uint32 size) = 0;

	virtual uint32 receive(void *buffer, uint32 size, uint32 timeOutMS) = 0;

	virtual void close() = 0;

	virtual bool isConnected() const = 0;

};
