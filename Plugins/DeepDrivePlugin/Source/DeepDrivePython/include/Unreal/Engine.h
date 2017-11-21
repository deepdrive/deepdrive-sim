
#pragma once

#include <stdint.h>
#include <string>

typedef int8_t int8;
typedef uint8_t uint8;
typedef int16_t int16;
typedef uint16_t uint16;
typedef int32_t int32;
typedef uint32_t uint32;

struct FVector
{
   float X = 0.0f;
   float Y = 0.0f;
   float Z = 0.0f;
};

class FString
{
public:

	FString(const char *str = 0)
		:	m_String(str ? str : "")
	{
	}

	FString(const std::string &str)
		:	m_String(str)
	{
	}

	const char* operator * () const
	{
		return m_String.c_str();
	}

private:

	std::string		m_String;
};

class FMemory
{
public:

	static void* Malloc(size_t, uint32 alignment = 0)
	{
		return 0;
	}

	static void Memcpy(void* , const void*, size_t)
	{
	}

};

template<typename T>
inline T* TCHAR_TO_ANSI (T *in)
{
	return in;
}
