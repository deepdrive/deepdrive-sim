#pragma once

#include "Engine.h"

namespace deepdrive { namespace utils {

void copyString(const char *src, char *dst, uint32 dstSize);

inline void copyString(const FName &src, char *dst, uint32 dstSize)
{
	copyString(TCHAR_TO_ANSI(*(src.ToString())), dst, dstSize);
}

} }
