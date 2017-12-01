// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "Engine.h"
#include "Runtime/Core/Public/HAL/Runnable.h"


DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveConnectionListener, Log, All);

/**
 * 
 */
class DeepDriveConnectionListener	:	public FRunnable
{
public:

	DeepDriveConnectionListener(uint8 a, uint8 b, uint8 c, uint8 d, uint16 port);

	~DeepDriveConnectionListener();

	virtual bool Init();
	virtual uint32 Run();
	virtual void Stop();

	void terminate();

private:

	void shutdown();

	FRunnableThread					*m_WorkerThread = 0;
	bool							m_isStopped = false;

	FSocket							*m_ListenSocket = 0;
	bool							m_isListening = false;
};
