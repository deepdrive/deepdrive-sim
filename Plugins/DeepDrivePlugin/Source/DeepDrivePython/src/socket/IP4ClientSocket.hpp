
#pragma once

#include "Engine.h"

class IIP4ClientSocketImpl;
struct IP4Address;

class IP4ClientSocket
{

public:

	IP4ClientSocket();

	~IP4ClientSocket();

	bool connect(const IP4Address &ip4Address);

	uint32 send(const void *data, uint32 bytesToSend);

	uint32 receive(void *buffer, uint32 size);

	bool receive(void *buffer, uint32 size, uint32 timeOutMS);

	void close();

	bool isConnected() const;

private:

	bool resizeReceiveBuffer(uint32 numBytes);

	IIP4ClientSocketImpl		*m_ClientSocketImpl;

	uint8						*m_ReceiveBuffer = 0;
	uint32						m_ReceiveBufferSize = 0;
	uint32						m_curWritePos = 0;

};
