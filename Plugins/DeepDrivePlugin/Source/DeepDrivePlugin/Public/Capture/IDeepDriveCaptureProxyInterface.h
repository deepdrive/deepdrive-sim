
#pragma once

#include "Engine.h"
#include "DeepDriveData.h"

class UCaptureSinkComponentBase;

class IDeepDriveCaptureProxyInterface
{
	
public:	

	virtual TArray<UCaptureSinkComponentBase*>& getSinks() = 0;

	virtual const DeepDriveDataOut& getDeepDriveData() const = 0;

};
