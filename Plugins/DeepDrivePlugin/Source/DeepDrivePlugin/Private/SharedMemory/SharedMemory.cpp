
#include "DeepDrivePluginPrivatePCH.h"

#include "Public/SharedMemory/SharedMemory.h"

#ifdef DEEPDRIVE_PLATFORM_LINUX
#include "Private/SharedMemory/SharedMemoryImpl_Linux.h"
#elif DEEPDRIVE_PLATFORM_WINDOWS
#include "Private/SharedMemory/SharedMemoryImpl_Windows.h"
#endif


SharedMemory::SharedMemory()
	: m_SharedMemImpl(0)
{
#ifdef DEEPDRIVE_PLATFORM_LINUX

	m_SharedMemImpl = new SharedMemoryImpl_Linux;

#elif DEEPDRIVE_PLATFORM_WINDOWS

	m_SharedMemImpl = new SharedMemoryImpl_Windows;

#endif
}

SharedMemory::~SharedMemory()
{
	delete m_SharedMemImpl;
}

bool SharedMemory::create(const FString &name, uint32 maxSize)
{
	m_maxSize = maxSize;
	return m_SharedMemImpl ? m_SharedMemImpl->create(name, maxSize) : false;
}


void* SharedMemory::lockForWriting(int32 waitTimeMS)
{
	return m_SharedMemImpl ? m_SharedMemImpl->lockForWriting(waitTimeMS) : 0;
}


void SharedMemory::unlock(uint32 size)
{
	if (m_SharedMemImpl)
		m_SharedMemImpl->unlock(size);
}

bool SharedMemory::tryConnect(const FString &name, uint32 maxSize)
{
	const bool connected = m_SharedMemImpl ? m_SharedMemImpl->tryConnect(name, maxSize) : false;
	if(connected)
		m_maxSize = maxSize;

	return connected;
}

bool SharedMemory::connect(const FString &name, uint32 maxSize)
{
	m_maxSize = maxSize;
	return m_SharedMemImpl ? m_SharedMemImpl->connect(name, maxSize) : false;
}

void SharedMemory::disconnect()
{
	if (m_SharedMemImpl)
		m_SharedMemImpl->disconnect();
}


const void* SharedMemory::lockForReading(int32 waitTimeMS) const
{
	return m_SharedMemImpl ? m_SharedMemImpl->lockForReading(waitTimeMS) : 0;
}


void SharedMemory::unlock()
{
	if (m_SharedMemImpl)
		m_SharedMemImpl->unlock();
}

int32 SharedMemory::getMaxPayloadSize() const
{
	return m_SharedMemImpl ? m_SharedMemImpl->getMaxPayloadSize() : 0;
}

