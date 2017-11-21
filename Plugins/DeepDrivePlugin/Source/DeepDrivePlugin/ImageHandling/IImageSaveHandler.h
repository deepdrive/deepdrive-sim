
#pragma once

/**
	Interface for saving images to file system
*/

namespace deepdrive
{

class Image;

class IImageSaveHandler
{
public:

	virtual bool save(const FString &fileName, const Image &img) = 0;

};

}	//	namespace
