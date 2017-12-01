// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "Engine.h"

namespace deepdrive { namespace server {
struct MessageHeader;
} }

DECLARE_DELEGATE_OneParam(HandleMessageDelegate, const deepdrive::server::MessageHeader&);

/**
 * 
 */
class DeepDriveMessageAssembler
{
public:

	DeepDriveMessageAssembler();

	~DeepDriveMessageAssembler();

	void add(const uint8 *data, uint32 numBytes);

public:

	HandleMessageDelegate		m_HandleMessage;

private:

	void checkForMessage();

	bool resize(uint32 numBytes);

	uint8						*m_MessageBuffer = 0;
	uint32						m_curBufferSize = 0;
	uint32						m_curWritePos = 0;

};
