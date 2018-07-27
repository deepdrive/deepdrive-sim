// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "Private/Server/DeepDriveConnectionThread.h"


DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveSimulationServer, Log, All);

/**
 * 
 */
class DeepDriveSimulationServer	:	public DeepDriveConnectionThread
{
public:

	DeepDriveSimulationServer(uint8 a, uint8 b, uint8 c, uint8 d, uint16 port);

	virtual ~DeepDriveSimulationServer();

	virtual bool Init();
	virtual uint32 Run();

	virtual float execute();

private:

	enum State
	{
		Idle,
		Listening,
		Connected
	};

	FSocket* listen();

	void checkForMessages();

	void shutdown();

	FSocket							*m_ListenSocket = 0;
	bool							m_isListening = false;

	FSocket							*m_Socket = 0;

	State							m_State = Idle;
};
