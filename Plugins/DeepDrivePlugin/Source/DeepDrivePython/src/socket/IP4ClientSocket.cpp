
#include "DeepDrivePluginPrivatePCH.h"

#include "socket/IP4ClientSocket.hpp"

#include <chrono>
#include <thread>
#include <iostream>
#include <stdlib.h>
#include <string.h>

#ifdef DEEPDRIVE_PLATFORM_LINUX
#include "socket/IP4ClientSocketImpl_Linux.hpp"
#elif DEEPDRIVE_PLATFORM_WINDOWS
#include "socket/IP4ClientSocketImpl_Windows.hpp"
#endif

IP4ClientSocket::IP4ClientSocket()
	:	m_ClientSocketImpl(0)
{

#ifdef DEEPDRIVE_PLATFORM_LINUX

	m_ClientSocketImpl = new IP4ClientSocketImpl_Linux();

#elif DEEPDRIVE_PLATFORM_WINDOWS

	m_ClientSocketImpl = new IP4ClientSocketImpl_Windows();

#endif

	(void) resizeReceiveBuffer(2 * 10240);

}

IP4ClientSocket::~IP4ClientSocket()
{
}

bool IP4ClientSocket::connect(const IP4Address &ip4Address)
{
	return m_ClientSocketImpl ? m_ClientSocketImpl->connect(ip4Address) : false;
}

uint32 IP4ClientSocket::send(const void *data, uint32 bytesToSend)
{
	return m_ClientSocketImpl ? m_ClientSocketImpl->send(data, bytesToSend) : 0;
}

uint32 IP4ClientSocket::receive(void *buffer, uint32 size)
{
	return m_ClientSocketImpl ? m_ClientSocketImpl->receive(buffer, size) : 0;
}

bool IP4ClientSocket::receive(void *buffer, uint32 expectedSize, uint32 timeOutMS)
{
	const uint32 readChunkSize = 2048;
	bool messageComplete = false;

	while(!messageComplete)
	{
		if(resizeReceiveBuffer(readChunkSize))
		{
			const uint32 bytesRead = m_ClientSocketImpl->receive(m_ReceiveBuffer + m_curWritePos, readChunkSize, timeOutMS);
			if(bytesRead)
			{
				m_curWritePos += bytesRead;
				if(m_curWritePos >= expectedSize)
				{
					memcpy(buffer, m_ReceiveBuffer, expectedSize);

					m_curWritePos -= expectedSize;
					if(m_curWritePos > 0)
					{
						memmove(m_ReceiveBuffer, m_ReceiveBuffer + expectedSize, m_curWritePos);
					}
					messageComplete = true;
					break;
				}
			}
			else
				break;
		}
	}
	return messageComplete;
}

void IP4ClientSocket::close()
{
	if(m_ClientSocketImpl)
		m_ClientSocketImpl->close();
}

bool IP4ClientSocket::isConnected() const
{
	return m_ClientSocketImpl ? m_ClientSocketImpl->isConnected() : false;
}


bool IP4ClientSocket::resizeReceiveBuffer(uint32 numBytes)
{
	const uint32 newSize = m_curWritePos + numBytes;
	if	(	m_ReceiveBuffer == 0
		||	m_ReceiveBufferSize < newSize
		)
	{
		m_ReceiveBuffer = reinterpret_cast<uint8*> (realloc(m_ReceiveBuffer, newSize));
		if(m_ReceiveBuffer)
			m_ReceiveBufferSize = newSize;
		else
			m_ReceiveBufferSize = 0;
	}
	return m_ReceiveBufferSize > 0;
}
