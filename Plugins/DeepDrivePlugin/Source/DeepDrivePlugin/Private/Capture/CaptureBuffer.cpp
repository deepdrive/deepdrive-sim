	
#include "Capture/CaptureBuffer.h"
#include "Capture/CaptureBufferPool.h"

CaptureBuffer::CaptureBuffer(CaptureBufferPool &captureBufferPool, EPixelFormat pixelFormat, uint32 width, uint32 height, uint32 stride)
	:	m_CaptureBufferPool(captureBufferPool)
	,	m_LockCounter(0)
	,	m_PixelFormat(pixelFormat)
	,	m_Width(width)
	,	m_Height(height)
	,	m_Stride(stride)
{
}

void CaptureBuffer::initialize(EPixelFormat pixelFormat, uint32 width, uint32 height, uint32 stride)
{
	m_PixelFormat = pixelFormat;
	m_Width = width;
	m_Height = height;
	m_Stride = stride;
}

bool CaptureBuffer::allocate()
{
	bool allocated = false;

	const uint32 bufferSize = m_Stride * m_Height;
	m_Buffer = FMemory::Malloc(bufferSize, 4);
	if(m_Buffer)
	{
		allocated = true;
		m_BufferSize = bufferSize;
	}
	return allocated;
}

void CaptureBuffer::release()
{
	if(m_SecondaryCaptureBuffer)
		m_SecondaryCaptureBuffer->release();

	m_LockCounter.Add(-1);
	if(m_LockCounter.GetValue() <= 0)
	{
		m_SecondaryCaptureBuffer = 0;
		m_CaptureBufferPool.release(*this);
	}
}

CaptureBuffer::DataType CaptureBuffer::getDataType() const
{
	DataType dataType = Undefined;

	if	(	m_PixelFormat == PF_FloatRGBA
		||	m_PixelFormat == PF_R16F
		)
	{
		dataType = Float16;
	}
	else if(m_PixelFormat == PF_B8G8R8A8)
	{
		dataType = UnsignedByte;
	}

	return dataType;
}

void CaptureBuffer::addLock()
{
	m_LockCounter.Add(1);
	if(m_SecondaryCaptureBuffer)
		m_SecondaryCaptureBuffer->addLock();
}
