// Fill out your copyright notice in the Description page of Project Settings.

#include "DeepDriveSimulationServer.h"

#include "Simulation/DeepDriveSimulation.h"
#include "Server/Messages/DeepDriveServerMessageHeader.h"

#include "Runtime/Networking/Public/Interfaces/IPv4/IPv4SubnetMask.h"
#include "Runtime/Networking/Public/Interfaces/IPv4/IPv4Address.h"
#include "Runtime/Sockets/Public/IPAddress.h"
#include "Runtime/Sockets/Public/SocketSubsystem.h"
#include "Runtime/Networking/Public/Interfaces/IPv4/IPv4Endpoint.h"
#include "Runtime/Sockets/Public/Sockets.h"
#include "Runtime/Networking/Public/Common/TcpSocketBuilder.h"

DEFINE_LOG_CATEGORY(LogDeepDriveSimulationServer);

DeepDriveSimulationServer::DeepDriveSimulationServer(ADeepDriveSimulation &simulation, int32 ipParts[4], uint16 port)
:	DeepDriveConnectionThread("DeepDriveSimulationServer")
,	m_Simulation(simulation)
{
	UE_LOG(LogDeepDriveSimulationServer, Log, TEXT("DeepDriveSimulationServer listening on %d.%d.%d.%d:%d"), ipParts[0], ipParts[1], ipParts[2], ipParts[3], port);

	FIPv4Endpoint endpoint(FIPv4Address(ipParts[0], ipParts[1], ipParts[2], ipParts[3]), port);
	m_ServerSocket = FTcpSocketBuilder(TEXT("DeepDriveSimulationServer_Listen")).AsReusable().BoundToEndpoint(endpoint).Listening(8);

	if(!m_ServerSocket)
		UE_LOG(LogDeepDriveSimulationServer, Log, TEXT("PANIC: Couldn't create Listening socket on %d.%d.%d.%d:%d"), ipParts[0], ipParts[1], ipParts[2], ipParts[3], port);

	resizeReceiveBuffer(10240);
}

DeepDriveSimulationServer::~DeepDriveSimulationServer()
{
	if(m_ClientSocket)
	{
		m_ClientSocket->Close();
		delete m_ClientSocket;
		UE_LOG(LogDeepDriveSimulationServer, Log, TEXT("Client socket closed") );
	}

	if(m_ServerSocket)
	{
		m_ServerSocket->Close();
		delete m_ServerSocket;
		UE_LOG(LogDeepDriveSimulationServer, Log, TEXT("Server socket closed") );
	}
}


bool DeepDriveSimulationServer::Init()
{
	UE_LOG(LogDeepDriveSimulationServer, Log, TEXT("DeepDriveSimulationServer::Init") );
	m_State = Listening;

	m_MessageAssembler.m_HandleMessage.BindRaw(this, &DeepDriveSimulationServer::handleMessage);

	return true;
}

uint32 DeepDriveSimulationServer::Run()
{
	UE_LOG(LogDeepDriveSimulationServer, Log, TEXT("DeepDriveSimulationServer Started"));

	while (!m_isStopped)
	{
		float sleepTime = 0.05f;
		switch(m_State)
		{
			case Idle:
				break;

			case Listening:
				m_ClientSocket = listen();
				if(m_ClientSocket)
				{
					m_State = Connected;
				}
				break;

			case Connected:
				if(checkForMessages())
				{
					deepdrive::server::MessageHeader *response = 0;
					if (	m_ResponseQueue.Dequeue(response)
						&&	response
						)
					{
						int32 bytesSent = 0;
						m_ClientSocket->Send(reinterpret_cast<uint8*> (response), response->message_size, bytesSent);
						UE_LOG(LogDeepDriveSimulationServer, Log, TEXT(" %d bytes sent back"), bytesSent);

						FMemory::Free(response);
						sleepTime = 0.001f;
					}
				}
				else
				{
					m_ClientSocket->Close();
					m_isListening = false;
					m_State = Listening;
					UE_LOG(LogDeepDriveSimulationServer, Log, TEXT("Connection to client lost, reverting back to listening") );
				}
				break;
		}
		FPlatformProcess::Sleep(sleepTime);
	}

	UE_LOG(LogDeepDriveSimulationServer, Log, TEXT("DeepDriveSimulationServer Finished"));

	shutdown();

	return 0;
}

void DeepDriveSimulationServer::enqueueResponse(deepdrive::server::MessageHeader *message)
{
	if (message)
	{
		m_ResponseQueue.Enqueue(message);
		UE_LOG(LogDeepDriveSimulationServer, Log, TEXT("enqueueResponse: Response queued"));
	}
}

FSocket* DeepDriveSimulationServer::listen()
{
	FSocket *socket = 0;
	if (m_ServerSocket)
	{
		if (!m_isListening)
		{
			m_isListening = m_ServerSocket->Listen(1);
			UE_LOG(LogDeepDriveSimulationServer, Log, TEXT("DeepDriveSimulationServer Socket is listening %c"), m_isListening ? 'T' : 'F');
		}

		bool pending = false;
		if (m_ServerSocket->HasPendingConnection(pending) && pending)
		{
			TSharedRef<FInternetAddr> remoteAddress = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->CreateInternetAddr();
			UE_LOG(LogDeepDriveSimulationServer, Log, TEXT("DeepDriveSimulationServer Incoming connection from %s"), *(remoteAddress->ToString(true)) );
			socket = m_ServerSocket->Accept(*remoteAddress, FString("DeepDriveSimulationServer"));
			if(socket)
				socket->SetNonBlocking(true);
		}
	}
	return socket;
}

bool DeepDriveSimulationServer::checkForMessages()
{
	int32 bytesRead = 0;
	bool connected = m_ClientSocket->Recv(m_ReceiveBuffer, m_curReceiveBufferSize, bytesRead, ESocketReceiveFlags::None);
	if (connected && bytesRead > 0)
	{
		if (bytesRead >= 8)
		{
			// uint32 *buffer = reinterpret_cast<uint32*>(m_ReceiveBuffer);
			// UE_LOG(LogDeepDriveSimulationServer, Log, TEXT("Received %d bytes: %d %d"), bytesRead, buffer[0], buffer[1]);
		}
		m_MessageAssembler.add(m_ReceiveBuffer, bytesRead);
	}

	return connected;
}

void DeepDriveSimulationServer::handleMessage(const deepdrive::server::MessageHeader &message)
{
	m_Simulation.enqueueMessage(message.clone());
}

void DeepDriveSimulationServer::shutdown()
{
	if (m_ServerSocket)
	{
		const bool success = m_ServerSocket->Close();
		UE_LOG(LogDeepDriveSimulationServer, Log, TEXT("DeepDriveSimulationServer::shutdown Socket successfully closed %c"), success ? 'T' : 'F');
		delete m_ServerSocket;
		m_ServerSocket = 0;
	}
}
