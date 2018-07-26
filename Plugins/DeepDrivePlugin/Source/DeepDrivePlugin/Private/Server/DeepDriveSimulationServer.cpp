// Fill out your copyright notice in the Description page of Project Settings.

#include "DeepDrivePluginPrivatePCH.h"
#include "DeepDriveSimulationServer.h"

#include "Private/Server/DeepDriveServer.h"

#include "Runtime/Networking/Public/Interfaces/IPv4/IPv4SubnetMask.h"
#include "Runtime/Networking/Public/Interfaces/IPv4/IPv4Address.h"
#include "Runtime/Sockets/Public/IPAddress.h"
#include "Runtime/Sockets/Public/SocketSubsystem.h"
#include "Runtime/Networking/Public/Interfaces/IPv4/IPv4Endpoint.h"
#include "Runtime/Sockets/Public/Sockets.h"
#include "Runtime/Networking/Public/Common/TcpSocketBuilder.h"

DEFINE_LOG_CATEGORY(LogDeepDriveSimulationServer);

DeepDriveSimulationServer::DeepDriveSimulationServer(uint8 a, uint8 b, uint8 c, uint8 d, uint16 port)
{
	UE_LOG(LogDeepDriveSimulationServer, Log, TEXT("DeepDriveSimulationServer listening on %d.%d.%d.%d:%d"), a, b, c, d, port);

	m_WorkerThread = FRunnableThread::Create(this, TEXT("DeepDriveSimulationServer") , 0, TPri_Normal);

	FIPv4Endpoint endpoint(FIPv4Address(a, b, c, d), port);
	m_ListenSocket = FTcpSocketBuilder(TEXT("DeepDriveSimulationServer_Listen")).AsReusable().BoundToEndpoint(endpoint).Listening(8);

	if(!m_ListenSocket)
		UE_LOG(LogDeepDriveSimulationServer, Log, TEXT("PANIC: Couldn't create Listening socket on %d.%d.%d.%d:%d"), a, b, c, d, port);
}

DeepDriveSimulationServer::~DeepDriveSimulationServer()
{
}


bool DeepDriveSimulationServer::Init()
{
	return true;
}

uint32 DeepDriveSimulationServer::Run()
{
	UE_LOG(LogDeepDriveSimulationServer, Log, TEXT("DeepDriveSimulationServer::Run Started") );

	m_State = Listening;

	while (!m_isStopped)
	{
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
				break;
		}

		FPlatformProcess::Sleep(0.05f);
	}

	UE_LOG(LogDeepDriveSimulationServer, Log, TEXT("DeepDriveSimulationServer::Run Finished"));

	shutdown();

	return 0;
}

FSocket* DeepDriveSimulationServer::listen()
{
	FSocket *socket = 0;
	if (m_ListenSocket)
	{
		if (!m_isListening)
		{
			m_isListening = m_ListenSocket->Listen(1);
			UE_LOG(LogDeepDriveSimulationServer, Log, TEXT("DeepDriveSimulationServer::Run Socket is listening %c"), m_isListening ? 'T' : 'F');
		}

		bool pending = false;
		if (m_ListenSocket->HasPendingConnection(pending) && pending)
		{
			TSharedRef<FInternetAddr> remoteAddress = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->CreateInternetAddr();
			socket = m_ListenSocket->Accept(*remoteAddress, FString("DeepDriveClient"));
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
			#if 0
			if (resizeReceiveBuffer(pendingSize))
			{
				int32 bytesRead = 0;
				if (m_Socket->Recv(m_ReceiveBuffer, m_curReceiveBufferSize, bytesRead, ESocketReceiveFlags::None))
				{
					// UE_LOG(LogDeepDriveClientConnection, Log, TEXT("[%d] Received %d bytes: %d"), m_ClientId, bytesRead, bytesRead > 4 ? *(reinterpret_cast<uint32*>(m_ReceiveBuffer)) : 0);
					m_MessageAssembler.add(m_ReceiveBuffer, bytesRead);
					sleepTime = 0.0f;
				}
			}
			#endif
		}
	}
}


void DeepDriveSimulationServer::Stop()
{
	UE_LOG(LogDeepDriveSimulationServer, Log, TEXT("DeepDriveSimulationServer::Stop"));
	shutdown();
}


void DeepDriveSimulationServer::terminate()
{
	m_isStopped = true;
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
