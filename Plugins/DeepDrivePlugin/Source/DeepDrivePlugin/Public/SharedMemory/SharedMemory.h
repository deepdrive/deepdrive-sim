#pragma once

#include "Engine.h"

class ISharedMemoryImpl;

class  SharedMemory
{
public:

	SharedMemory();
	~SharedMemory();

	/**
	writing
	*/
	bool create(const FString &name, uint32 maxSize);

	void* lockForWriting(int32 waitTimeMS);

	void unlock(uint32 size);



	/**
	reading
	*/
	bool tryConnect(const FString &name, uint32 maxSize);

	bool connect(const FString &name, uint32 maxSize);

	void disconnect();

	const void* lockForReading(int32 waitTimeMS) const;

	void unlock();

	int32 getMaxPayloadSize() const;

private:

	ISharedMemoryImpl			*m_SharedMemImpl;

	uint32						m_maxSize = 0;

};
