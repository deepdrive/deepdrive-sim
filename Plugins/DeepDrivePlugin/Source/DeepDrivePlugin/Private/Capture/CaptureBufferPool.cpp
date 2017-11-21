
#include "DeepDrivePluginPrivatePCH.h"
#include "Private/Capture/CaptureBufferPool.h"
#include "Private/Capture/CaptureBuffer.h"

DEFINE_LOG_CATEGORY(LogCaptureBufferPool);

CaptureBuffer* CaptureBufferPool::acquire(EPixelFormat pixelFormat, uint32 width, uint32 height, uint32 stride)
{
	CaptureBuffer *captureBuffer = 0;
	const uint32 bufferSize = stride * height;

	m_Mutex.Lock();
	if(!m_BufferSlots.Contains(bufferSize))
	{
		m_BufferSlots.Add(bufferSize, SBufferSlot());
	}

	SBufferSlot &bufferSlot = m_BufferSlots[bufferSize];
	if(bufferSlot.available_buffers.Num() > 0)
	{
		captureBuffer = bufferSlot.available_buffers.Pop();
		bufferSlot.used_buffers.Push(captureBuffer);

		captureBuffer->initialize(pixelFormat, width, height, stride);
	}
	else
	{
		captureBuffer = new CaptureBuffer(*this, pixelFormat, width, height, stride);
		if	(	captureBuffer
			&&	captureBuffer->allocate()
			)
		{
			bufferSlot.allocated_buffers.Push(captureBuffer);
			bufferSlot.used_buffers.Push(captureBuffer);
		}
		logMemorySize();
	}

	m_Mutex.Unlock();


	return captureBuffer;
}

void CaptureBufferPool::release(CaptureBuffer &buffer)
{
	const int32 bufferSize = buffer.getBufferSize();

	m_Mutex.Lock();
	if(m_BufferSlots.Contains(bufferSize))
	{
		SBufferSlot &bufferSlot = m_BufferSlots[bufferSize];

		const int32 ind = bufferSlot.used_buffers.Find(&buffer);
		if(ind != INDEX_NONE)
		{
			bufferSlot.used_buffers.RemoveAt(ind);
			bufferSlot.available_buffers.Push(&buffer);
		}
	}
	m_Mutex.Unlock();
}

void CaptureBufferPool::logMemorySize()
{
	uint32 totalSize = 0;
	for (auto &slot : m_BufferSlots)
	{
		const uint32 curSize = slot.Key * slot.Value.allocated_buffers.Num();
		totalSize += curSize;
		UE_LOG(LogCaptureBufferPool, Log, TEXT("SlotSize %d Allocated %d"), slot.Key, curSize);
	}
	UE_LOG(LogCaptureBufferPool, Log, TEXT("Total size %d"), totalSize);
}
