// Fill out your copyright notice in the Description page of Project Settings.

#include "DeepDrivePluginPrivatePCH.h"
#include "DeepDriveSimulationServer.h"

#include "Public/Simulation/DeepDriveSimulation.h"
#include "Public/Server/Messages/DeepDriveServerMessageHeader.h"

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
	m_ListenSocket = FTcpSocketBuilder(TEXT("DeepDriveSimulationServer_Listen")).AsReusable().BoundToEndpoint(endpoint).Listening(8);

	if(!m_ListenSocket)
		UE_LOG(LogDeepDriveSimulationServer, Log, TEXT("PANIC: Couldn't create Listening socket on %d.%d.%d.%d:%d"), ipParts[0], ipParts[1], ipParts[2], ipParts[3], port);
}

DeepDriveSimulationServer::~DeepDriveSimulationServer()
{
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
				m_Socket = listen();
				if(m_Socket)
				{
					m_State = Connected;
				}
				break;

			case Connected:
				checkForMessages();
				{
					deepdrive::server::MessageHeader *response = 0;
					if (m_ResponseQueue.Dequeue(response)
						&& response
						)
					{
						int32 bytesSent = 0;
						m_Socket->Send(reinterpret_cast<uint8*> (response), response->message_size, bytesSent);
						// UE_LOG(LogDeepDriveClientConnection, Log, TEXT("[%d] %d bytes sent back"), m_ClientId, bytesSent);

						FMemory::Free(response);
						sleepTime = 0.001f;
					}

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
}

FSocket* DeepDriveSimulationServer::listen()
{
	FSocket *socket = 0;
	if (m_ListenSocket)
	{
		if (!m_isListening)
		{
			m_isListening = m_ListenSocket->Listen(1);
			UE_LOG(LogDeepDriveSimulationServer, Log, TEXT("DeepDriveSimulationServer Socket is listening %c"), m_isListening ? 'T' : 'F');
		}

		bool pending = false;
		if (m_ListenSocket->HasPendingConnection(pending) && pending)
		{
			TSharedRef<FInternetAddr> remoteAddress = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->CreateInternetAddr();
			UE_LOG(LogDeepDriveSimulationServer, Log, TEXT("DeepDriveSimulationServer Incoming connection from %s"), *(remoteAddress->ToString(true)) );
			socket = m_ListenSocket->Accept(*remoteAddress, FString("DeepDriveSimulationServer"));
		}
	}
	return socket;
}

void DeepDriveSimulationServer::checkForMessages()
{
	uint32 pendingSize = 0;
	if (m_Socket->HasPendingData(pendingSize))
	{
		if (pendingSize)
		{
			if (resizeReceiveBuffer(pendingSize))
			{
				int32 bytesRead = 0;
				if (m_Socket->Recv(m_ReceiveBuffer, m_curReceiveBufferSize, bytesRead, ESocketReceiveFlags::None))
				{
					// UE_LOG(LogDeepDriveClientConnection, Log, TEXT("[%d] Received %d bytes: %d"), m_ClientId, bytesRead, bytesRead > 4 ? *(reinterpret_cast<uint32*>(m_ReceiveBuffer)) : 0);
					m_MessageAssembler.add(m_ReceiveBuffer, bytesRead);
				}
			}
		}
	}
}

void DeepDriveSimulationServer::handleMessage(const deepdrive::server::MessageHeader &message)
{
	m_Simulation.enqueueMessage(message.clone());
}

void DeepDriveSimulationServer::shutdown()
{
	if (m_ListenSocket)
	{
		const bool success = m_ListenSocket->Close();
		UE_LOG(LogDeepDriveSimulationServer, Log, TEXT("DeepDriveSimulationServer::shutdown Socket successfully closed %c"), success ? 'T' : 'F');
		delete m_ListenSocket;
		m_ListenSocket = 0;
	}
}
