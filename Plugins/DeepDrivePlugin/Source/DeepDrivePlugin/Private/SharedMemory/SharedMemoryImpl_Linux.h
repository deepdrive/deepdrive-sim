
#pragma once

#include "Private/SharedMemory/ISharedMemoryImpl.h"

#ifdef DEEPDRIVE_PLATFORM_LINUX

#ifdef DEEPDRIVE_WITH_UE4_LOGGING
DECLARE_LOG_CATEGORY_EXTERN(LogSharedMemoryImpl_Linux, Log, All);
#endif

class SharedMemoryImpl_Linux	:	public ISharedMemoryImpl
{

	struct SharedMemoryData
	{
		// MEMORY MANAGEMENT ------------------
		unsigned long long		version_read = 0;
		unsigned long long		version_written = 0;
		uint32					is_client_connected = false;

		pthread_mutex_t			mutex;
		uint32					mutex_created_successfully = false;

		char					data[1];
	};


public:

	SharedMemoryImpl_Linux();
	virtual ~SharedMemoryImpl_Linux();

	/**
		writing
	*/
	virtual bool create(const FString &name, uint32 maxSize);

	virtual void* lockForWriting(int32 waitTimeMS);

	virtual void unlock(uint32 size);



	/**
		reading
	*/
	virtual bool tryConnect(const FString &name, uint32 maxSize);

	virtual bool connect(const FString &name, uint32 maxSize);

	virtual void disconnect();

	virtual const void* lockForReading(int32 waitTimeMS) const;

	virtual void unlock();

	virtual int32 getMaxPayloadSize() const;

private:

	bool connect_Impl(const FString &name, uint32 maxSize);

	SharedMemoryData* createSharedMem(const char *name, uint32 maxSize);

	SharedMemoryData* openSharedMem(const char *name, uint32 maxSize);

	void createMutex();


	OperationMode					m_OperationMode = OperationMode::Undefined;

	FString							m_SharedMemName;
	mutable SharedMemoryData		*m_SharedMemoryData = 0;
	uint32							m_maxSize = 0;

	pthread_mutexattr_t				m_MutexAttr;

	mutable bool					m_isLocked = false;

	bool							m_reportConnectionErrors = true;

};

#endif
