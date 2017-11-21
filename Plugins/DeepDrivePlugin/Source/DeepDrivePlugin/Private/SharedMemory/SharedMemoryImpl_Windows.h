
#pragma once

#ifdef DEEPDRIVE_PLATFORM_WINDOWS

#include "Private/SharedMemory/ISharedMemoryImpl.h"


#include <windows.h>
#include <future>
#include <sstream>



#ifdef DEEPDRIVE_WITH_UE4_LOGGING
	DECLARE_LOG_CATEGORY_EXTERN(LogSharedMemoryImpl_Windows, Log, All);
#endif

class SharedMemoryImpl_Windows : public ISharedMemoryImpl
{
	enum OperationMode
	{
		Undefined,
		Read,
		Write
	};


	struct SharedMemoryData
	{
		// MEMORY MANAGEMENT ------------------
		unsigned long long		version_read = 0;
		unsigned long long		version_written = 0;
		uint32					is_client_connected = 0;

		uint32					mutex_created_successfully = 0;

		char					data[1];
	};


public:

	SharedMemoryImpl_Windows();
	virtual ~SharedMemoryImpl_Windows();

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


	SharedMemoryData* createSharedMem(const TCHAR *name, uint32 maxSize);

	SharedMemoryData* openSharedMem(const TCHAR *name, uint32 maxSize);

	void createMutex(const char *name);


	OperationMode					m_OperationMode = OperationMode::Undefined;

	FString							m_SharedMemName;
	mutable SharedMemoryData		*m_SharedMemoryData = 0;
	uint32							m_maxSize = 0;

	int32							m_MutexId;
	mutable bool					m_isLocked = false;

	HANDLE							m_FileMap;

	HANDLE							m_Mutex = 0;

	bool							m_reportConnectionErrors = false;
};

#endif
