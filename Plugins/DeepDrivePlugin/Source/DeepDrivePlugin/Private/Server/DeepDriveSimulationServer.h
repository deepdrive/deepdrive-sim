// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "Private/Server/DeepDriveConnectionThread.h"


DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveSimulationServer, Log, All);

namespace deepdrive { namespace server {
struct MessageHeader;
} }

class ADeepDriveSimulation;

/**
 * 
 */
class DeepDriveSimulationServer	:	public DeepDriveConnectionThread
{
public:

	DeepDriveSimulationServer(ADeepDriveSimulation &simulation, int32 ipParts[4], uint16 port);

	virtual ~DeepDriveSimulationServer();

	virtual bool Init();
	virtual uint32 Run();

	void enqueueResponse(deepdrive::server::MessageHeader *message);

private:

	enum State
	{
		Idle,
		Listening,
		Connected
	};

	FSocket* listen();

	void checkForMessages();

	void handleMessage(const deepdrive::server::MessageHeader &message);

	void shutdown();

	ADeepDriveSimulation			&m_Simulation;
	FSocket							*m_ListenSocket = 0;
	bool							m_isListening = false;

	FSocket							*m_Socket = 0;

	State							m_State = Idle;

	typedef TQueue<deepdrive::server::MessageHeader*, EQueueMode::Mpsc>	MessageQueue;
	MessageQueue						m_ResponseQueue;
};
