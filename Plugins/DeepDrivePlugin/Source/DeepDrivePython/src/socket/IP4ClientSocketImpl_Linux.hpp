
#pragma once

#include "socket/IIP4ClientSocketImpl.hpp"

class IP4ClientSocketImpl_Linux	:	public IIP4ClientSocketImpl
{

public:

	IP4ClientSocketImpl_Linux();
	~IP4ClientSocketImpl_Linux();

	virtual bool connect(const IP4Address &ip4Address);

	virtual uint32 send(const void *data, uint32 bytesToSend);

	virtual uint32 receive(void *buffer, uint32 size);

	virtual uint32 receive(void *buffer, uint32 size, uint32 timeOutMS);

	virtual void close();

	virtual bool isConnected() const;

private:

	int32				m_SocketHandle = 0;

	bool				m_isConnected = false;
};
