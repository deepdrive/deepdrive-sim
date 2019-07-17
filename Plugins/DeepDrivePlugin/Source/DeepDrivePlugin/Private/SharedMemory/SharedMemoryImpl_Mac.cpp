
#ifdef DEEPDRIVE_PLATFORM_MAC


#include "DeepDrivePluginPrivatePCH.h"
#include "Private/SharedMemory/SharedMemoryImpl_Mac.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <fcntl.h>
#include <pthread.h>
#include <iostream>
#include <errno.h>
#include <sys/stat.h>
#include <sstream>


#ifdef DEEPDRIVE_WITH_UE4_LOGGING
DEFINE_LOG_CATEGORY(LogSharedMemoryImpl_Mac);
#else

#include <iostream>

#endif

SharedMemoryImpl_Mac::SharedMemoryImpl_Mac()
{
}

SharedMemoryImpl_Mac::~SharedMemoryImpl_Mac()
{
    disconnect();
}

bool SharedMemoryImpl_Mac::create(const FString &name, uint32 maxSize)
{
    bool created = false;
    if(m_OperationMode == OperationMode::Undefined)
    {
        m_SharedMemoryData = createSharedMem(TCHAR_TO_ANSI(*name), maxSize);

        if(m_SharedMemoryData)
        {
            m_SharedMemName = name;
            m_maxSize = maxSize;
            m_OperationMode = OperationMode::Write;
            created = true;

#ifdef DEEPDRIVE_WITH_UE4_LOGGING
            UE_LOG(LogSharedMemoryImpl_Mac, Log, TEXT("SharedMemoryImpl_Mac::create Shared mem %s with size %d successfully created at %p  sizeof bool %d  sizeof %d"), *(name), maxSize, m_SharedMemoryData, sizeof(bool), sizeof(SharedMemoryData));
#endif
            createMutex();
        }
    }

    return created;
}

void* SharedMemoryImpl_Mac::lockForWriting(int32 waitTimeMS)
{
    void *res = 0;
    if(!m_isLocked && m_SharedMemoryData)
    {
        if(waitTimeMS == 0)
            m_isLocked = pthread_mutex_trylock(&m_SharedMemoryData->mutex) == 0;
        else
            m_isLocked = pthread_mutex_lock(&m_SharedMemoryData->mutex) == 0;

        if(m_isLocked)
        {
            res = &m_SharedMemoryData->data;
#ifdef DEEPDRIVE_WITH_UE4_LOGGING
            UE_LOG(LogSharedMemoryImpl_Mac, Verbose, TEXT("Locked for writing"));
#else
//			std::cout << "Locked for writing\n";
#endif
        }
        else
        {
#ifdef DEEPDRIVE_WITH_UE4_LOGGING
            UE_LOG(LogSharedMemoryImpl_Mac, Verbose, TEXT("NOT LOCKED FOR WRITING"));
#else
//			std::cout << "NOT LOCKED FOR WRITING\n";
#endif
        }
    }

    return res;
}

void SharedMemoryImpl_Mac::unlock(uint32 size)
{
    if(m_isLocked && m_SharedMemoryData)
        pthread_mutex_unlock (&m_SharedMemoryData->mutex);

    m_isLocked = false;
}

bool SharedMemoryImpl_Mac::tryConnect(const FString &name, uint32 maxSize)
{
    m_reportConnectionErrors = false;
    return connect_Impl(name, maxSize);
}

bool SharedMemoryImpl_Mac::connect(const FString &name, uint32 maxSize)
{
    m_reportConnectionErrors = true;
    return connect_Impl(name, maxSize);
}

bool SharedMemoryImpl_Mac::connect_Impl(const FString &name, uint32 maxSize)
{
    bool connected = false;

    if(m_OperationMode == OperationMode::Undefined)
    {
        m_SharedMemoryData = openSharedMem(TCHAR_TO_ANSI(*name), maxSize);

        if(m_SharedMemoryData)
        {
            m_SharedMemName = name;
            m_maxSize = maxSize;
            m_OperationMode = OperationMode::Read;
            connected = true;

            if(m_reportConnectionErrors)
            {
#ifdef DEEPDRIVE_WITH_UE4_LOGGING
                UE_LOG(LogSharedMemoryImpl_Mac, Log, TEXT("SharedMemoryImpl_Linux::connect Connected successfully to shared mem %s with size %d"), *(name), maxSize);
#endif
            }
        }
    }

    return connected;
}

void SharedMemoryImpl_Mac::disconnect()
{
    if (m_OperationMode != OperationMode::Undefined)
    {
        if (m_OperationMode == OperationMode::Write)
        {
#ifdef DEEPDRIVE_WITH_UE4_LOGGING
            UE_LOG(LogSharedMemoryImpl_Mac, Display, TEXT("SharedMemoryImpl_Linux Destroying inter-process mutex"));
#else

            std::cout << "SharedMemoryImpl_Linux Destroying inter-process mutex\n";
#endif

            int destroyRes = pthread_mutex_destroy(&m_SharedMemoryData->mutex);
            if (destroyRes != 0)
            {
#ifdef DEEPDRIVE_WITH_UE4_LOGGING
                UE_LOG(LogSharedMemoryImpl_Mac, Display,
				    TEXT("SharedMemoryImpl_Linux Error, failed to destroy inter-process mutex. Errno: %d"), destroyRes);
#else
                std::cout << "SharedMemoryImpl_Linux Error, failed to destroy inter-process mutex. Errno: " << destroyRes << "\n";
#endif
            }
        }

        if (munmap(m_SharedMemoryData, sizeof(m_maxSize)) == -1)
        {
#ifdef DEEPDRIVE_WITH_UE4_LOGGING
            UE_LOG(LogSharedMemoryImpl_Mac, Error, TEXT("SharedMemoryImpl_Linux unmapping failed"));
#else
            std::cout << "SharedMemoryImpl_Linux unmapping failed\n";
#endif
        }

        if (m_OperationMode == OperationMode::Write)
        {
            unlink(TCHAR_TO_ANSI(*m_SharedMemName));
        }
    }

    m_OperationMode = OperationMode::Undefined;
    m_SharedMemoryData = 0;
}

const void* SharedMemoryImpl_Mac::lockForReading(int32 waitTimeMS) const
{
    void *res = 0;
    if(!m_isLocked && m_SharedMemoryData)
    {
        if(waitTimeMS == 0)
            m_isLocked = pthread_mutex_trylock(&m_SharedMemoryData->mutex) == 0;
        else
            m_isLocked = pthread_mutex_lock(&m_SharedMemoryData->mutex) == 0;

        if(m_isLocked)
        {
            res = &m_SharedMemoryData->data;
#ifdef DEEPDRIVE_WITH_UE4_LOGGING
            UE_LOG(LogSharedMemoryImpl_Mac, Verbose, TEXT("Locked for reading"));
#else
//            std::cout << "Locked for reading\n";
#endif
        }
        else
        {
#ifdef DEEPDRIVE_WITH_UE4_LOGGING
            UE_LOG(LogSharedMemoryImpl_Mac, Verbose, TEXT("NOT LOCKED FOR READING"));
#else
            std::cout << "NOT LOCKED FOR READING\n";
#endif
        }
    }
#ifndef DEEPDRIVE_WITH_UE4_LOGGING
    else
        std::cout << "Is Locked or no shared mem" << m_isLocked << "\n";
#endif

    return res;
}

void SharedMemoryImpl_Mac::unlock()
{
    if(m_isLocked && m_SharedMemoryData)
        pthread_mutex_unlock (&m_SharedMemoryData->mutex);

    m_isLocked = false;
}

int32 SharedMemoryImpl_Mac::getMaxPayloadSize() const
{
    return m_maxSize - sizeof(SharedMemoryData);
}



/*
 * get_shared_mem - creates a memory mapped file area.
 * The return value is a page-aligned memory value, or NULL if there is a failure.
 * Here's the list of arguments:
 * @mmapFileName - the name of the memory mapped file
 * @size - the size of the memory mapped file (should be a multiple of the system page for best performance)
 * @create - determines whether or not the area should be created.
 */
SharedMemoryImpl_Mac::SharedMemoryData* SharedMemoryImpl_Mac::createSharedMem(const char *name, uint32 maxSize)
{
    void *ret = 0;

    mode_t origMask = umask(0);
    int32 mmapFd = open(name, O_CREAT | O_RDWR, 00666);
    umask(origMask);
    if (mmapFd < 0)
    {
#ifdef DEEPDRIVE_WITH_UE4_LOGGING
        UE_LOG(LogSharedMemoryImpl_Mac, Error, TEXT("SharedMemoryImpl_Linux::createSharedMem Open mmapFd failed"));
#endif
        return 0;
    }

    if ((ftruncate(mmapFd, maxSize) == 0))
    {
        int result = (int) lseek(mmapFd, maxSize - 1, SEEK_SET);
        if (result == -1)
        {
#ifdef DEEPDRIVE_WITH_UE4_LOGGING
            UE_LOG(LogSharedMemoryImpl_Mac, Error, TEXT("SharedMemoryImpl_Linux::createSharedMem lseek failed"));
#endif
            close(mmapFd);
            return 0;
        }

        /* Something needs to be written at the end of the file to
        * have the file actually have the new size.
        * Just writing an empty string at the current file position will do.
        * Note:
        *  - The current position in the file is at the end of the stretched
        *    file due to the call to lseek().
        *  - The current position in the file is at the end of the stretched
        *    file due to the call to lseek().
        *  - An empty string is actually a single '\0' character, so a zero-byte
        *    will be written at the last byte of the file.
        */
        result = (int) write(mmapFd, "", 1);
        if (result != 1)
        {
#ifdef DEEPDRIVE_WITH_UE4_LOGGING
            UE_LOG(LogSharedMemoryImpl_Mac, Error, TEXT("SharedMemoryImpl_Linux::createSharedMem write mmapFd failed"));
#endif
            close(mmapFd);
            return 0;
        }

        ret = mmap(NULL, maxSize, PROT_READ | PROT_WRITE, MAP_SHARED, mmapFd, 0);

        if (ret == MAP_FAILED || ret == NULL)
        {
            close(mmapFd);
            return 0;
        }
    }

    return reinterpret_cast<SharedMemoryData*> (ret);
}

SharedMemoryImpl_Mac::SharedMemoryData* SharedMemoryImpl_Mac::openSharedMem(const char *name, uint32 maxSize)
{
    void *ret = 0;

    int mmapFd = open(name, O_RDWR, 00666);
    if (mmapFd < 0)
    {
        return NULL;
    }

    int result = (int) lseek(mmapFd, 0, SEEK_END);
    if (result == -1)
    {
        if(m_reportConnectionErrors)
        {
#ifdef DEEPDRIVE_WITH_UE4_LOGGING
            UE_LOG(LogSharedMemoryImpl_Mac, Error, TEXT("SharedMemoryImpl_Mac::openSharedMem lseek mmapFd failed"));
#endif
        }
        close(mmapFd);
        return NULL;
    }

    if (result == 0)
    {
        if(m_reportConnectionErrors)
        {
#ifdef DEEPDRIVE_WITH_UE4_LOGGING
            UE_LOG(LogSharedMemoryImpl_Mac, Error, TEXT("SharedMemoryImpl_Mac::openSharedMem he file has 0 bytes"));
#endif
        }
        close(mmapFd);
        return NULL;
    }

    ret = mmap(NULL, maxSize, PROT_READ | PROT_WRITE, MAP_SHARED, mmapFd, 0);

    if (ret == MAP_FAILED || ret == NULL)
    {
        if(m_reportConnectionErrors)
        {
#ifdef DEEPDRIVE_WITH_UE4_LOGGING
            UE_LOG(LogSharedMemoryImpl_Mac, Error, TEXT("SharedMemoryImpl_Mac::openSharedMem mmap failed"));
#endif
        }
        close(mmapFd);
        return NULL;
    }

    close(mmapFd);

    return reinterpret_cast<SharedMemoryData*> (ret);
}

void SharedMemoryImpl_Mac::createMutex()
{
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

    // initialize a mutex attribute
    if(pthread_mutexattr_init(&m_MutexAttr) == -1)
    {
#ifdef DEEPDRIVE_WITH_UE4_LOGGING
        UE_LOG(LogSharedMemoryImpl_Mac, Error, TEXT("SharedMemoryImpl_Mac::createMutex pthread_mutexattr_init() failed"));
#endif
        return;
    }

    // Allow sharing the mutex across processes
    if(pthread_mutexattr_setpshared(&m_MutexAttr, PTHREAD_PROCESS_SHARED) == -1)
    {
#ifdef DEEPDRIVE_WITH_UE4_LOGGING
        UE_LOG(LogSharedMemoryImpl_Mac, Error, TEXT("SharedMemoryImpl_Mac::createMutex pthread_mutexattr_setpshared() failed"));
#endif
        return;
    }

    if(pthread_mutex_init(&mutex, &m_MutexAttr) == -1)
    {
#ifdef DEEPDRIVE_WITH_UE4_LOGGING
        UE_LOG(LogSharedMemoryImpl_Mac, Error, TEXT("SharedMemoryImpl_Mac::createMutex pthread_mutex_init() failed"));
#endif
        return;
    }

    m_SharedMemoryData->mutex = mutex;
    m_SharedMemoryData->mutex_created_successfully = true;
};

#endif