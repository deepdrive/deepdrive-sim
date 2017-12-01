// Fill out your copyright notice in the Description page of Project Settings.

#include "DeepDrivePluginPrivatePCH.h"
#include "DeepDriveConnectionListener.h"

#include "Private/Server/DeepDriveServer.h"

#include "Runtime/Networking/Public/Interfaces/IPv4/IPv4SubnetMask.h"
#include "Runtime/Networking/Public/Interfaces/IPv4/IPv4Address.h"
#include "Runtime/Sockets/Public/IPAddress.h"
#include "Runtime/Sockets/Public/SocketSubsystem.h"
#include "Runtime/Networking/Public/Interfaces/IPv4/IPv4Endpoint.h"
#include "Runtime/Sockets/Public/Sockets.h"
#include "Runtime/Networking/Public/Common/TcpSocketBuilder.h"

DEFINE_LOG_CATEGORY(LogDeepDriveConnectionListener);

DeepDriveConnectionListener::DeepDriveConnectionListener(uint8 a, uint8 b, uint8 c, uint8 d, uint16 port)
{
	m_WorkerThread = FRunnableThread::Create(this, TEXT("DeepDriveConnectionListener") , 0, TPri_Normal);

	UE_LOG(LogDeepDriveConnectionListener, Log, TEXT("Listening on %d.%d.%d.%d:%d"), a, b, c, d, port);

	FIPv4Endpoint endpoint(FIPv4Address(a, b, c, d), port);
	m_ListenSocket = FTcpSocketBuilder(TEXT("DeepDriverServer_Listen")).AsReusable().BoundToEndpoint(endpoint).Listening(8);

	if(!m_ListenSocket)
		UE_LOG(LogDeepDriveConnectionListener, Log, TEXT("PANIC: Couldn't create Listening socket on %d.%d.%d.%d:%d"), a, b, c, d, port);
}

DeepDriveConnectionListener::~DeepDriveConnectionListener()
{
}


bool DeepDriveConnectionListener::Init()
{
	return true;
}

uint32 DeepDriveConnectionListener::Run()
{
	UE_LOG(LogDeepDriveConnectionListener, Log, TEXT("DeepDriveConnectionListener::Run Started") );

	while (!m_isStopped)
	{
		if (m_ListenSocket)
		{
			if (!m_isListening)
			{
				m_isListening = m_ListenSocket->Listen(10);
				UE_LOG(LogDeepDriveConnectionListener, Log, TEXT("DeepDriveConnectionListener::Run Socket is listening %c"), m_isListening ? 'T' : 'F');
			}

			bool pending = false;
			if (m_ListenSocket->HasPendingConnection(pending) && pending)
			{
				TSharedRef<FInternetAddr> remoteAddress = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->CreateInternetAddr();
				FSocket *socket = m_ListenSocket->Accept(*remoteAddress, FString("DeepDriveClient"));
				if (socket)
				{
					DeepDriveServer::GetInstance().addIncomingConnection(socket, remoteAddress);
				}
			}
		}

		FPlatformProcess::Sleep(0.05f);
	}

	UE_LOG(LogDeepDriveConnectionListener, Log, TEXT("DeepDriveConnectionListener::Run Finished"));

	shutdown();

	return 0;
}

void DeepDriveConnectionListener::Stop()
{
	UE_LOG(LogDeepDriveConnectionListener, Log, TEXT("DeepDriveConnectionListener::Stop"));
	shutdown();
}


void DeepDriveConnectionListener::terminate()
{
	m_isStopped = true;
}

void DeepDriveConnectionListener::shutdown()
{
	if (m_ListenSocket)
	{
		const bool success = m_ListenSocket->Close();
		UE_LOG(LogDeepDriveConnectionListener, Log, TEXT("DeepDriveConnectionListener::shutdown Socket successfully closed %c"), success ? 'T' : 'F');
		delete m_ListenSocket;
		m_ListenSocket = 0;
	}
}
