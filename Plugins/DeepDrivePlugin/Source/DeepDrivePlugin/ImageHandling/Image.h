
#pragma once

namespace deepdrive
{

class Image
{
public:

	enum Format
	{
		Undefined,
		RGB,
		RGBA,
		Greyscale
	};

	~Image();

	bool allocate(Format format, uint32 width, uint32 height);

	void storeAsRGB(const uint8 *src, uint32 width, uint32 height);
	void storeAsRGB(const TArray<FColor> &src, uint32 width, uint32 height);
	void storeAsRGB(const TArray<FLinearColor> &src, uint32 width, uint32 height);
	void storeAsRGB(const FFloat16 *src, uint32 width, uint32 height);
	void storeAsRGBA(const TArray<FColor> &src, uint32 width, uint32 height);
	void storeAsGreyscale(const TArray<FColor> &src, uint32 width, uint32 height);
	void storeAsGreyscale(const FFloat16 *src, uint32 width, uint32 height);

	uint32 getWidth() const;
	uint32 getHeight() const;

	uint32 getNumComponents() const;
	uint32 getSizeInBytes() const;

	template<class T>
	T* getRawPtr();

	template<class T>
	const T* getRawPtr() const;

private:

	void init(Format format, uint32 width, uint32 height, uint32 bytesPerComponent = 1);

	uint32				m_Width = 0;
	uint32				m_Height = 0;

	Format				m_Format = Undefined;

	uint8				*m_Data = 0;

	uint32				m_SizeInBytes;
};


inline 	uint32 Image::getWidth() const
{
	return m_Width;
}

inline 	uint32 Image::getHeight() const
{
	return m_Height;
}

inline 	uint32 Image::getSizeInBytes() const
{
	return m_SizeInBytes;
}

template<class T>
inline T* Image::getRawPtr()
{
	return reinterpret_cast<const T*> (m_Data);
}

template<class T>
inline const T* Image::getRawPtr() const
{
	return reinterpret_cast<const T*> (m_Data);
}

}
