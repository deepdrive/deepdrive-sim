
#include "DeepDrivePluginPrivatePCH.h"
#include "ImageHandling/BmpSaveHandler.h"
#include "ImageHandling/Image.h"

namespace deepdrive
{


BmpSaveHandler::BmpSaveHandler()
{
	
}

BmpSaveHandler::~BmpSaveHandler()
{
	
}


bool BmpSaveHandler::save(const FString &fileName, const Image &img)
{
	bool res = false;

	const int32 fileSize = static_cast<int> (img.getSizeInBytes());
	const int32 width = static_cast<int> (img.getWidth());
	const int32 height = static_cast<int> (img.getHeight());

	FILE *out = fopen(TCHAR_TO_ANSI(*fileName), "wb");
	if (out)
	{

		SBmpFileMagic bm = { {'B', 'M'} };
		SBmpFileHeader bh = { 54 + fileSize, 0, 0, 54 };
		SBitmapInfoHeader bmpInfoHeader = { 40, width, height, 1, 24, 0, 0, 0, 0, 0, 0 };

		fwrite(&bm, 1, sizeof(bm), out);
		fwrite(&bh, 1, sizeof(bh), out);
		fwrite(&bmpInfoHeader, 1, sizeof(bmpInfoHeader), out);

		const uint8 *data = img.getRawPtr<uint8>();
		const uint32 numPadding = (4 - (width * 3) % 4) % 4;
		if (numPadding == 0)
		{
			for (signed i = height - 1; i >= 0; --i)
				fwrite(data + (width * i * 3), 3, width, out);
		}
		else
		{
			char bmpPadBytes[3] = { 0, 0, 0 };
			for (signed i = 0; i < height; ++i)
			{
				fwrite(data + (width * (height - i - 1) * 3), 3, width, out);
				if (numPadding)
					fwrite(bmpPadBytes, 1, numPadding, out);
			}
		}
		fclose(out);
		res = true;
	}

	return res;
}


}	//	namespace
