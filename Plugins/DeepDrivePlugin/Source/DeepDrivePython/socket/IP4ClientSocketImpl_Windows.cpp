
#pragma once

#include "socket/IP4ClientSocketImpl_Windows.hpp"
#include "socket/IP4Address.hpp"

#include <iostream>
 
#pragma comment(lib,"ws2_32.lib") //Winsock Library

IP4ClientSocketImpl_Windows::IP4ClientSocketImpl_Windows()
{
	WSADATA wsa;

	std::cout << "Initialising Winsock...\n";
	if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0)
	{
		std::cout << "Failed. Error Code : " << WSAGetLastError() << "\n";
	}

	std::cout << "Initialised.\n";

}

IP4ClientSocketImpl_Windows::~IP4ClientSocketImpl_Windows()
{
	close();
}

bool IP4ClientSocketImpl_Windows::connect(const IP4Address &ip4Address)
{
	m_Socket = socket(AF_INET , SOCK_STREAM , 0 );
	if(m_Socket != INVALID_SOCKET)
	{
		std::cout << "Socket created\n";
		sockaddr_in serverAddress;

		memset(&serverAddress, 0, sizeof(serverAddress));
		serverAddress.sin_family = AF_INET;
		serverAddress.sin_addr.s_addr = inet_addr( ip4Address.toStr(false).c_str());
		serverAddress.sin_port = htons(ip4Address.port);

		if (::connect(m_Socket, reinterpret_cast<const sockaddr*> (&serverAddress), sizeof(serverAddress)) == 0)
		{
			m_isConnected = true;
			std::cout << "Connected\n";
		}
		else
			std::cout << "Connectinbg failed. Error Code : " << WSAGetLastError() << "\n";

	}
	else
		std::cout << "Creating socket failed\n";
	return false;
}

uint32 IP4ClientSocketImpl_Windows::send(const void *data, uint32 bytesToSend)
{
	uint32 bytesSend = 0;
	if(m_isConnected)
	{
		int32 res = ::send(m_Socket, reinterpret_cast<const char*> (data), bytesToSend, 0);
		if(res >= 0)
		{
			bytesSend = static_cast<uint32> (res);
			std::cout << "Sent " << bytesSend << " bytes\n";
		}
		else
			std::cout << "Send failed. Error Code : " << WSAGetLastError() << "\n";
	}
	return bytesSend;
}

uint32 IP4ClientSocketImpl_Windows::receive(void *buffer, uint32 size)
{
	uint32 res = 0;
	if(m_isConnected)
	{
		int32 receivedSize = ::recv(m_Socket, reinterpret_cast<char*> (buffer), size , 0);

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

uint32 IP4ClientSocketImpl_Windows::receive(void *buffer, uint32 size, uint32 timeOutMS)
{
	uint32 res = 0;

	fd_set readFds;
	readFds.fd_count = 1;
	readFds.fd_array[0] = m_Socket;

	timeval timeOut;
	timeOut.tv_sec = 0;
	timeOut.tv_usec = timeOutMS * 1000;

	int32 numReady = ::select(0, &readFds, 0, 0, &timeOut);
	if(numReady > 0)
	{
		int32 receivedSize = ::recv(m_Socket, reinterpret_cast<char*> (buffer), size, 0);

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


void IP4ClientSocketImpl_Windows::close()
{
	if(m_Socket != INVALID_SOCKET)
	{
		closesocket(m_Socket);
		m_Socket = INVALID_SOCKET;
		std::cout << "Socket closed\n";
	}
	WSACleanup();
}

bool IP4ClientSocketImpl_Windows::isConnected() const
{
	return m_isConnected;
}
