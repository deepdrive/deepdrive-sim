
#ifdef DEEPDRIVE_PLATFORM_WINDOWS

#include "DeepDrivePluginPrivatePCH.h"
#include "Private/SharedMemory/SharedMemoryImpl_Windows.h"

#ifdef DEEPDRIVE_WITH_UE4_LOGGING
DEFINE_LOG_CATEGORY(LogSharedMemoryImpl_Windows);
#else
#include <iostream>
#endif

SharedMemoryImpl_Windows::SharedMemoryImpl_Windows()
{
}

SharedMemoryImpl_Windows::~SharedMemoryImpl_Windows()
{
	disconnect();
}

bool SharedMemoryImpl_Windows::create(const FString &name, uint32 maxSize)
{
	bool created = false;
	if (m_OperationMode == OperationMode::Undefined)
	{
		m_SharedMemoryData = createSharedMem(*name, maxSize);
		if (m_SharedMemoryData)
		{
			//m_SharedMemName = name;
			m_maxSize = maxSize;
			m_OperationMode = OperationMode::Write;
			created = true;

#ifdef DEEPDRIVE_WITH_UE4_LOGGING
			UE_LOG(LogSharedMemoryImpl_Windows, Log, TEXT("SharedMemoryImpl_Windows::create Shared mem %s with size %d successfully created at %p"), *(name), maxSize, m_SharedMemoryData);
#else
			std::cout << "SharedMemoryImpl_Windows::create Shared mem " << *name << " with size " << maxSize << " successfully created\n";
#endif

			m_Mutex = CreateMutex(0, false, TEXT ("DeepDriveCaptureSnapshotMutex"));
		}
	}
	return created;
}

void* SharedMemoryImpl_Windows::lockForWriting(int32 waitTimeMS)
{
	void *res = 0;
	if (!m_isLocked && m_SharedMemoryData)
	{
		int32 lockRes = WaitForSingleObject(m_Mutex, waitTimeMS < 0 ? INFINITE : waitTimeMS);
		if (lockRes == 0)
		{
			m_isLocked = true;
			res = &m_SharedMemoryData->data;
		}
	}

	return res;
}

void SharedMemoryImpl_Windows::unlock(uint32 size)
{
	if (m_isLocked && m_SharedMemoryData)
	{
		int32 releaseRes = ReleaseMutex(m_Mutex);
		m_isLocked = releaseRes == 0 ? true : false;
	}
}


bool SharedMemoryImpl_Windows::tryConnect(const FString &name, uint32 maxSize)
{
	m_reportConnectionErrors = false;
	return connect_Impl(name, maxSize);
}

bool SharedMemoryImpl_Windows::connect(const FString &name, uint32 maxSize)
{
	m_reportConnectionErrors = true;
	return connect_Impl(name, maxSize);
}

bool SharedMemoryImpl_Windows::connect_Impl(const FString &name, uint32 maxSize)
{
	bool connected = false;

	if (m_OperationMode == OperationMode::Undefined)
	{
		m_SharedMemoryData = openSharedMem(*name, maxSize);

		if (m_SharedMemoryData)
		{
			m_maxSize = maxSize;
			m_OperationMode = OperationMode::Read;
			connected = true;

#ifdef DEEPDRIVE_WITH_UE4_LOGGING
			UE_LOG(LogSharedMemoryImpl_Windows, Log, TEXT("SharedMemoryImpl_Windows::connect Connected successfully to shared mem %s with size %d\n"), *name, maxSize);
#else
			std::cout << "SharedMemoryImpl_Windows::connect Connected successfully to shared mem " << *name << " with size " << maxSize << "\n";
#endif

			SECURITY_ATTRIBUTES secAttribs;
			secAttribs.nLength = sizeof(secAttribs);
			secAttribs.lpSecurityDescriptor = NULL;
			secAttribs.bInheritHandle = true;
			m_Mutex = CreateMutex(&secAttribs, true, TEXT("DeepDriveCaptureSnapshotMutex"));

		}
	}

	return connected;
}

void SharedMemoryImpl_Windows::disconnect()
{
	if (m_SharedMemoryData && m_FileMap)
	{
#ifdef DEEPDRIVE_WITH_UE4_LOGGING
		UE_LOG(LogSharedMemoryImpl_Windows, Log, TEXT("SharedMemoryImpl_Windows::disconnect Disconnecting\n"));
#else
		std::cout << "SharedMemoryImpl_Windows::disconnect Disconnecting\n";
#endif
		UnmapViewOfFile(m_SharedMemoryData);
		CloseHandle(m_FileMap);

		m_FileMap = 0;
		m_SharedMemoryData = 0;
	}
	else
	{
#ifdef DEEPDRIVE_WITH_UE4_LOGGING
		UE_LOG(LogSharedMemoryImpl_Windows, Log, TEXT("SharedMemoryImpl_Windows::disconnect Nothing to disconnect %p %p\n"), m_SharedMemoryData, m_FileMap);
#else
		std::cout << "SharedMemoryImpl_Windows::disconnect Nothing to disconnect " << static_cast<void*> (m_SharedMemoryData) << "  " << static_cast<void*> (m_FileMap) << "\n";
#endif
	}

	if (m_Mutex)
		CloseHandle(m_Mutex);
	m_Mutex = 0;

	m_OperationMode = OperationMode::Undefined;
}

const void* SharedMemoryImpl_Windows::lockForReading(int32 waitTimeMS) const
{
	void *res = 0;
	if (!m_isLocked && m_SharedMemoryData)
	{
		int32 lockRes = WaitForSingleObject(m_Mutex, waitTimeMS < 0 ? INFINITE : waitTimeMS);
		if (lockRes == 0)
		{
			m_isLocked = true;
			res = &m_SharedMemoryData->data;
		}
	}
	return res;
}

void SharedMemoryImpl_Windows::unlock()
{
	if (m_isLocked && m_SharedMemoryData)
	{
		ReleaseMutex(m_Mutex);
		m_isLocked = false;
	}
}

int32 SharedMemoryImpl_Windows::getMaxPayloadSize() const
{
	return m_maxSize - sizeof(SharedMemoryData);
}

SharedMemoryImpl_Windows::SharedMemoryData* SharedMemoryImpl_Windows::createSharedMem(const TCHAR *name, uint32 maxSize)
{
	m_FileMap =	CreateFileMapping	(
											INVALID_HANDLE_VALUE,    // use paging file
											NULL,                    // default security
											PAGE_READWRITE,          // read/write access
											0,                       // maximum object size (high-order DWORD)
											maxSize,                // maximum object size (low-order DWORD)
											name);                 // name of mapping object

	if (m_FileMap == NULL)
	{
#ifdef DEEPDRIVE_WITH_UE4_LOGGING
		UE_LOG(LogSharedMemoryImpl_Windows, Error, TEXT("SharedMemoryImpl_Windows::createSharedMem Could not create file mapping object (%d)"), GetLastError() );
#endif
		return 0;
	}

	SharedMemoryData *sharedMemData = reinterpret_cast<SharedMemoryData*> (MapViewOfFile(m_FileMap, FILE_MAP_ALL_ACCESS, 0, 0, maxSize));

	if (sharedMemData == NULL)
	{
#ifdef DEEPDRIVE_WITH_UE4_LOGGING
		UE_LOG(LogSharedMemoryImpl_Windows, Error, TEXT("SharedMemoryImpl_Windows::createSharedMem Could not map view of file (%d)"), GetLastError());
#endif
		CloseHandle(m_FileMap);
	}

	return sharedMemData;
}

SharedMemoryImpl_Windows::SharedMemoryData*SharedMemoryImpl_Windows::openSharedMem(const TCHAR *name, uint32 maxSize)
{
	m_FileMap = OpenFileMapping(FILE_MAP_ALL_ACCESS, false, name);

	if (m_FileMap == NULL)
	{
		if (m_reportConnectionErrors)
		{
#ifdef DEEPDRIVE_WITH_UE4_LOGGING
			UE_LOG(LogSharedMemoryImpl_Windows, Error, TEXT("SharedMemoryImpl_Windows::createSharedMem Could not open file mapping object (%d)"), GetLastError());
#else
			std::cout << "SharedMemoryImpl_Windows::openSharedMem SharedMemoryImpl_Windows::createSharedMem Could not open file mapping object " << GetLastError() << "\n";
#endif
		}
		return 0;
	}

	SharedMemoryData *sharedMemData = reinterpret_cast<SharedMemoryData*> (MapViewOfFile(m_FileMap, FILE_MAP_ALL_ACCESS, 0, 0, maxSize));

	if (sharedMemData == NULL)
	{
		if (m_reportConnectionErrors)
		{
#ifdef DEEPDRIVE_WITH_UE4_LOGGING
			UE_LOG(LogSharedMemoryImpl_Windows, Error, TEXT("SharedMemoryImpl_Windows::createSharedMem Could not map view of file (%d)"), GetLastError());
#else
			std::cout << "SharedMemoryImpl_Windows::createSharedMem Could not map view of file " << GetLastError() << "\n";
#endif
			CloseHandle(m_FileMap);
		}
	}

	return sharedMemData;
}

void SharedMemoryImpl_Windows::createMutex(const char *name)
{

}

#endif
