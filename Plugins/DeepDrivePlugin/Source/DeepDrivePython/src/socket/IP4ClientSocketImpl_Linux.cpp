
#include "common/ClientErrorCode.hpp"

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

int32 IP4ClientSocketImpl_Linux::send(const void *data, uint32 bytesToSend)
{
	int32 res = ClientErrorCode::NOT_CONNECTED;
	if(m_isConnected)
	{
		ssize_t bytesSent = ::write(m_SocketHandle, data, bytesToSend);
		if(bytesSent < 0)
		{
			if(errno == EPIPE)
			{
				// no valid connection anymore
				std::cout << "Connection lost\n";
				close();
				res = ClientErrorCode::CONNECTION_LOST;
			}
			else
				res = ClientErrorCode::UNKNOWN_ERROR;
		}
		else
			res = static_cast<int32> (bytesSent);
	}
	return res;
}

int32 IP4ClientSocketImpl_Linux::receive(void *buffer, uint32 size)
{
	int32 res = ClientErrorCode::NOT_CONNECTED;
	if(m_isConnected)
	{
		int32 receivedSize = ::recv(m_SocketHandle, reinterpret_cast<char*> (buffer), size , 0);

		if(receivedSize > 0)
		{
			res = receivedSize;
			// std::cout << "Received " << res << " bytes\n";
		}
		else
		{
			if	(	receivedSize == 0
				||	errno == EPIPE
				)
			{
				// no valid connection anymore
				std::cout << "Connection lost\n";
				close();
				res = ClientErrorCode::CONNECTION_LOST;
			}
			else
				res = ClientErrorCode::UNKNOWN_ERROR;
		}
    }
	return res;
}

int32 IP4ClientSocketImpl_Linux::receive(void *buffer, uint32 size, uint32 timeOutMS)
{
	int32 res = ClientErrorCode::NOT_CONNECTED;

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
			res = receivedSize;
			// std::cout << "Received " << res << " bytes\n";
		}
		else
		{
			if	(	receivedSize == 0
				||	errno == EPIPE
				)
			{
				// no valid connection anymore
				std::cout << "Connection lost\n";
				close();
				res = ClientErrorCode::CONNECTION_LOST;
			}
			else
				res = ClientErrorCode::UNKNOWN_ERROR;
		}
	}
	else
	{
		std::cout << "Poll timed out after " << timeOutMS << " msecs\n";
		res = ClientErrorCode::TIME_OUT;
	}

	return res;
}

void IP4ClientSocketImpl_Linux::close()
{
	if(m_isConnected)
	{
		::close(m_SocketHandle);
		m_SocketHandle = 0;
		m_isConnected = false;
		std::cout << "Socket closed\n";
	}
}

bool IP4ClientSocketImpl_Linux::isConnected() const
{
	return m_isConnected;
}
