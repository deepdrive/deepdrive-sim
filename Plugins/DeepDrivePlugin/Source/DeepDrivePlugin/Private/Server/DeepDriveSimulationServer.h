// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "Engine.h"
#include "Runtime/Core/Public/HAL/Runnable.h"


DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveSimulationServer, Log, All);

/**
 * 
 */
class DeepDriveSimulationServer	:	public FRunnable
{
public:

	DeepDriveSimulationServer(uint8 a, uint8 b, uint8 c, uint8 d, uint16 port);

	~DeepDriveSimulationServer();

	virtual bool Init();
	virtual uint32 Run();
	virtual void Stop();

	void terminate();

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

	FRunnableThread					*m_WorkerThread = 0;
	bool							m_isStopped = false;

	FSocket							*m_ListenSocket = 0;
	bool							m_isListening = false;

	FSocket							*m_Socket = 0;

	State							m_State = Idle;
};
