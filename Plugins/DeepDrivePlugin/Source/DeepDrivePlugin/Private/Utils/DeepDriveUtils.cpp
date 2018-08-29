
#include "DeepDrivePluginPrivatePCH.h"
#include "Private/Utils/DeepDriveUtils.h"

namespace deepdrive { namespace utils {

void copyString(const char *src, char *dst, uint32 dstSize)
{
	for (uint32 i = dstSize - 1; i > 0 && *src; *dst++ = *src++, --i);
	*dst = 0;
}

} }
