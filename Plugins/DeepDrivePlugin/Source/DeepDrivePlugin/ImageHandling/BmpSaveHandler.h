
#pragma once

#include "IImageSaveHandler.h"

/**
	Simple BmpFileSaver
*/

namespace deepdrive
{


class BmpSaveHandler	:	public IImageSaveHandler
{
	struct SBmpFileMagic
	{
		unsigned char magic[2];
	};

	struct SBmpFileHeader
	{
		int filesz;
		short creator1;
		short creator2;
		int bmp_offset;
	};

	struct SBitmapInfoHeader
	{
		int header_sz;
		int width;
		int height;
		short nplanes;
		short bitspp;
		int compress_type;
		int bmp_bytesz;
		int hres;
		int vres;
		int ncolors;
		int nimpcolors;
	};

public:

	BmpSaveHandler();
	~BmpSaveHandler();

	virtual bool save(const FString &fileName, const Image &img);

	
};


}	//	namespace

