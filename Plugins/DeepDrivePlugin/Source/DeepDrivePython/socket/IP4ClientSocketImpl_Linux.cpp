
#include "socket/IP4ClientSocketImpl_Linux.hpp"
#include "socket/IP4Address.hpp"

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h> 
#include <unistd.h>
#include <poll.h>

#include <arpa/inet.h>
#include <string.h>

#include <iostream>

IP4ClientSocketImpl_Linux::IP4ClientSocketImpl_Linux()
{

}

IP4ClientSocketImpl_Linux::~IP4ClientSocketImpl_Linux()
{
	close();
}

bool IP4ClientSocketImpl_Linux::connect(const IP4Address &ip4Address)
{
	if(!m_isConnected)
	{
		m_SocketHandle = socket(AF_INET, SOCK_STREAM, 0);
		if(m_SocketHandle)
		{
			sockaddr_in serverAddress;

			memset(&serverAddress, 0, sizeof(serverAddress));
			serverAddress.sin_family = AF_INET;
			inet_aton( ip4Address.toStr(false).c_str(), &serverAddress.sin_addr);
			serverAddress.sin_port = htons(ip4Address.port);

			if (::connect(m_SocketHandle, reinterpret_cast<const sockaddr*> (&serverAddress), sizeof(serverAddress)) == 0)
			{
				m_isConnected = true;
			}
		}
	}
	
	return m_isConnected;
}

uint32 IP4ClientSocketImpl_Linux::send(const void *data, uint32 bytesToSend)
{
	uint32 bytesSend = 0;
	if(m_isConnected)
	{
		ssize_t res = ::write(m_SocketHandle, data, bytesToSend);
		if(res >= 0)
			bytesSend = static_cast<uint32> (res);
	}
	return bytesSend;
}

uint32 IP4ClientSocketImpl_Linux::receive(void *buffer, uint32 size)
{
	uint32 res = 0;
	if(m_isConnected)
	{
		int32 receivedSize = ::recv(m_SocketHandle, reinterpret_cast<char*> (buffer), size , 0);

		if(receivedSize > 0)
		{
			res = static_cast<uint32> (receivedSize);
			std::cout << "Received " << res << " bytes\n";
		}
		else
			std::cout << "Received nothing " << receivedSize << "\n";
    }
	return res;
}

uint32 IP4ClientSocketImpl_Linux::receive(void *buffer, uint32 size, uint32 timeOutMS)
{
	uint32 res = 0;

	pollfd pollInOut;
	pollInOut.fd = m_SocketHandle;
	pollInOut.events = POLLIN;
	pollInOut.revents = 0;
	int32 pollRes = ::poll(&pollInOut, 1, timeOutMS);
	if(pollRes > 0)
	{
		int32 receivedSize = ::recv(m_SocketHandle, reinterpret_cast<char*> (buffer), size, 0);

		if(receivedSize > 0)
		{
			res = static_cast<uint32> (receivedSize);
			std::cout << "Received " << res << " bytes\n";
		}
		else
			std::cout << "Received nothing " << receivedSize << "\n";
	}
	else
		std::cout << "Poll timed out\n";

	return res;
}

void IP4ClientSocketImpl_Linux::close()
{
	if(m_isConnected)
	{
		::close(m_SocketHandle);
		m_SocketHandle = 0;
		m_isConnected = false;
	}
}

bool IP4ClientSocketImpl_Linux::isConnected() const
{
	return m_isConnected;
}
