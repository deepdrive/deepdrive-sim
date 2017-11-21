
#pragma once

#include "Engine.h"

class ISharedMemoryImpl
{
protected:

	enum OperationMode
	{
		Undefined,
		Read,
		Write
	};

public:

	virtual ~ISharedMemoryImpl()
		{	}

	/**
		writing
	*/
	virtual bool create(const FString &name, uint32 maxSize) = 0;

	virtual void* lockForWriting(int32 waitTimeMS) = 0;

	virtual void unlock(uint32 size) = 0;



	/**
		reading
	*/
	virtual bool tryConnect(const FString &name, uint32 maxSize) = 0;

	virtual bool connect(const FString &name, uint32 maxSize) = 0;

	virtual void disconnect() = 0;

	virtual const void* lockForReading(int32 waitTimeMS) const = 0;

	virtual void unlock() = 0;


	virtual int32 getMaxPayloadSize() const = 0;

};

