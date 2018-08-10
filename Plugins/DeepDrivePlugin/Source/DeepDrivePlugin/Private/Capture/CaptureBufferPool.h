
#pragma once

#include "Engine.h"

DECLARE_LOG_CATEGORY_EXTERN(LogCaptureBufferPool, Log, All);

class CaptureBuffer;

class CaptureBufferPool
{
	friend class CaptureBuffer;

	typedef TArray<CaptureBuffer*>	Buffers;

	struct SBufferSlot
	{
		Buffers			allocated_buffers;
		Buffers			available_buffers;
		Buffers			used_buffers;
	};

public:

	CaptureBuffer* acquire(EPixelFormat pixelFormat, uint32 width, uint32 height, uint32 stride);

private:

	void release(CaptureBuffer &buffer);

	void logMemorySize();

	FCriticalSection				m_Mutex;

	TMap<uint32, SBufferSlot>		m_BufferSlots;

};
