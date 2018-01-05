
#pragma once

#include "socket/IIP4ClientSocketImpl.hpp"

#include<winsock2.h>

class IP4ClientSocketImpl_Windows	:	public IIP4ClientSocketImpl
{

public:

	IP4ClientSocketImpl_Windows();
	~IP4ClientSocketImpl_Windows();

	virtual bool connect(const IP4Address &ip4Address);

	virtual uint32 send(const void *data, uint32 bytesToSend);

	virtual uint32 receive(void *buffer, uint32 size);

	virtual uint32 receive(void *buffer, uint32 size, uint32 timeOutMS);

	virtual void close();

	virtual bool isConnected() const;

private:

	SOCKET					m_Socket = INVALID_SOCKET;

	bool					m_isConnected = false;
};
