
#include "DeepDrivePluginPrivatePCH.h"
#include "Image.h"

namespace deepdrive
{

Image::~Image()
{
	delete[] m_Data;
}

void Image::init(Format format, uint32 width, uint32 height, uint32 bytesPerComponent)
{
	m_Format = format;
	m_Width = width;
	m_Height = height;
	m_SizeInBytes = width * height * getNumComponents() * bytesPerComponent;
}

bool Image::allocate(Format format, uint32 width, uint32 height)
{
	bool res = false;

	delete m_Data;

	init(format, width, height);
	m_Data = new uint8[m_SizeInBytes];

	return res;
}

void Image::storeAsRGB(const uint8 *src, uint32 width, uint32 height)
{
	init(RGB, width, height);
	m_Data = new uint8[m_SizeInBytes];

	uint8 *ptr = reinterpret_cast<uint8*> (m_Data);
	for(unsigned y = 0; y < height; ++y)
	{
		for(unsigned x = 0; x < width; ++x)
		{
			*ptr++ = src[0];
			*ptr++ = src[1];
			*ptr++ = src[2];
			src += 4;
		}
	}
}

void Image::storeAsRGB(const TArray<FColor> &src, uint32 width, uint32 height)
{
	init(RGB, width, height);
	m_Data = new uint8[m_SizeInBytes];

	uint8 *ptr = reinterpret_cast<uint8*> (m_Data);
	for (const FColor &col : src)
	{
		*ptr++ = col.B;
		*ptr++ = col.G;
		*ptr++ = col.R;
	}
}

void Image::storeAsRGB(const TArray<FLinearColor> &src, uint32 width, uint32 height)
{
	init(RGB, width, height);
	m_Data = new uint8[m_SizeInBytes];

	uint8 *ptr = reinterpret_cast<uint8*> (m_Data);
	for (const FLinearColor &col : src)
	{
		*ptr++ = static_cast<uint8> (col.B * 255.0f);
		*ptr++ = static_cast<uint8> (col.G * 255.0f);
		*ptr++ = static_cast<uint8> (col.R * 255.0f);
	}
}

void Image::storeAsRGB(const FFloat16 *src, uint32 width, uint32 height)
{
	init(RGB, width, height, 2);
	m_Data = new uint8[m_SizeInBytes];

	uint8 *ptr = reinterpret_cast<uint8*> (m_Data);
	for(unsigned y = 0; y < height; ++y)
	{
		for(unsigned x = 0; x < width; ++x)
		{
#if 1

			*ptr++ = static_cast<uint8> (FMath::Clamp( static_cast<float> (pow(src[2].GetFloat(), 0.45f)), 0.0f, 1.0f) * 255.0f);
			*ptr++ = static_cast<uint8> (FMath::Clamp( static_cast<float> (pow(src[1].GetFloat(), 0.45f)), 0.0f, 1.0f) * 255.0f);
			*ptr++ = static_cast<uint8> (FMath::Clamp( static_cast<float> (pow(src[0].GetFloat(), 0.45f)), 0.0f, 1.0f) * 255.0f);

			// *ptr++ = static_cast<uint8> (FMath::Clamp(src[2].GetFloat(), 0.0f, 1.0f) * 255.0f);
			// *ptr++ = static_cast<uint8> (FMath::Clamp(src[2].GetFloat(), 0.0f, 1.0f) * 255.0f);
			// *ptr++ = static_cast<uint8> (FMath::Clamp(src[2].GetFloat(), 0.0f, 1.0f) * 255.0f);

#else
			*ptr++ = static_cast<uint8> (FMath::Clamp(src[2].GetFloat(), 0.0f, 1.0f) * 255.0f);
			*ptr++ = static_cast<uint8> (FMath::Clamp(src[2].GetFloat(), 0.0f, 1.0f) * 255.0f);
			*ptr++ = static_cast<uint8> (FMath::Clamp(src[2].GetFloat(), 0.0f, 1.0f) * 255.0f);
#endif
			src += 4;
		}
	}
}

void Image::storeAsGreyscale(const FFloat16 *src, uint32 width, uint32 height)
{
	init(RGB, width, height, 2);
	m_Data = new uint8[m_SizeInBytes];

	uint8 *ptr = reinterpret_cast<uint8*> (m_Data);
	for(unsigned y = 0; y < height; ++y)
	{
		for(unsigned x = 0; x < width; ++x)
		{
			*ptr++ = static_cast<uint8> (FMath::Clamp(src[3].GetFloat() / 65535.0f, 0.0f, 1.0f) * 255.0f);
			*ptr++ = static_cast<uint8> (FMath::Clamp(src[3].GetFloat() / 65535.0f, 0.0f, 1.0f) * 255.0f);
			*ptr++ = static_cast<uint8> (FMath::Clamp(src[3].GetFloat() / 65535.0f, 0.0f, 1.0f) * 255.0f);
			src += 4;
		}
	}
}


void Image::storeAsRGBA(const TArray<FColor> &src, uint32 width, uint32 height)
{
	m_Format = RGBA;
	m_Width = width;
	m_Height = height;
	m_SizeInBytes = width * height * getNumComponents();
	m_Data = new uint8[m_SizeInBytes];

	uint8 *ptr = reinterpret_cast<uint8*> (m_Data);
	for (const FColor &col : src)
	{
		*ptr++ = col.G;
		*ptr++ = col.B;
		*ptr++ = col.R;
		*ptr++ = col.A;
	}
}

void Image::storeAsGreyscale(const TArray<FColor> &src, uint32 width, uint32 height)
{
}

uint32 Image::getNumComponents() const
{
	uint32 numComps[] = {0,3,4,1};
	return numComps[m_Format];
}



}	//	namespace
