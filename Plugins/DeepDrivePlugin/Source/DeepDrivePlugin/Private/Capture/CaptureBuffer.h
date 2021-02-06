
#pragma once

#include "Engine.h"
#include "Simulation/DeepDriveSimulationDefines.h"

class CaptureBufferPool;

class CaptureBuffer
{
public:

	enum DataType
	{
		Undefined,
		UnsignedByte,
		Float16,
		Float32
	};

	CaptureBuffer(CaptureBufferPool &captureBufferPool, EPixelFormat pixelFormat, uint32 width, uint32 height, uint32 stride);

	void initialize(EPixelFormat pixelFormat, uint32 width, uint32 height, uint32 stride);

	bool allocate();

	void addLock();
	void release();

	void setSecondaryCaptureBuffer(CaptureBuffer *buffer);
	CaptureBuffer* getSecondaryCaptureBuffer();

	void setEncoding(EDeepDriveInternalCaptureEncoding encoding);
	EDeepDriveInternalCaptureEncoding getEncoding();

	template<class T>
	T* getBuffer();
	template<class T>
	const T* getBuffer() const;

	EPixelFormat getPixelFormat() const;
	uint32 getWidth() const;
	uint32 getHeight() const;
	uint32 getStride() const;
	uint32 getBufferSize() const;

	DataType getDataType() const;

private:

	CaptureBufferPool						&m_CaptureBufferPool;

	FThreadSafeCounter						m_LockCounter;

	CaptureBuffer							*m_SecondaryCaptureBuffer = 0;

	EDeepDriveInternalCaptureEncoding		m_Encoding = EDeepDriveInternalCaptureEncoding::RGB_DEPTH;

	void									*m_Buffer = 0;
	EPixelFormat							m_PixelFormat = PF_Unknown;
	uint32									m_Width = 0;
	uint32									m_Height = 0;
	uint32									m_Stride = 0;
	uint32									m_BufferSize = 0;

};

inline void CaptureBuffer::setSecondaryCaptureBuffer(CaptureBuffer *buffer)
{
	m_SecondaryCaptureBuffer = buffer;
}

inline CaptureBuffer* CaptureBuffer::getSecondaryCaptureBuffer()
{
	return m_SecondaryCaptureBuffer;
}

inline void CaptureBuffer::setEncoding(EDeepDriveInternalCaptureEncoding encoding)
{
	m_Encoding = encoding;
}

inline EDeepDriveInternalCaptureEncoding CaptureBuffer::getEncoding()
{
	return m_Encoding;
}


template<class T>
inline T* CaptureBuffer::getBuffer()
{
	return reinterpret_cast<T*> (m_Buffer);
}

template<class T>
inline const T* CaptureBuffer::getBuffer() const
{
	return reinterpret_cast<const T*> (m_Buffer);
}

inline EPixelFormat CaptureBuffer::getPixelFormat() const
{
	return m_PixelFormat;
}

inline uint32 CaptureBuffer::getWidth() const
{
	return m_Width;
}

inline uint32 CaptureBuffer::getHeight() const
{
	return m_Height;
}

inline uint32 CaptureBuffer::getStride() const
{
	return m_Stride;
}

inline uint32 CaptureBuffer::getBufferSize() const
{
	return m_BufferSize;
}
