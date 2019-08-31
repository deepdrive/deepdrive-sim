
#include "DeepDrivePluginPrivatePCH.h"
#include "Private/Utils/DeepDriveUtils.h"

namespace deepdrive { namespace utils {

void copyString(const char *src, char *dst, uint32 dstSize)
{
	for (uint32 i = dstSize - 1; i > 0 && *src; *dst++ = *src++, --i);
	*dst = 0;
}

void expandBox2D(FBox2D &box, const FVector2D &p)
{
	box.Min.X = FMath::Min(box.Min.X, p.X);
	box.Min.Y = FMath::Min(box.Min.Y, p.Y);
	box.Max.X = FMath::Max(box.Max.X, p.X);
	box.Max.Y = FMath::Max(box.Max.Y, p.Y);
}


} }
