
#pragma once

#include "Engine.h"
#include "Public/DeepDriveData.h"

class UCaptureSinkComponentBase;

class IDeepDriveCaptureProxy
{
	
public:	

	virtual TArray<UCaptureSinkComponentBase*>& getSinks() = 0;

	virtual const DeepDriveDataOut& getDeepDriveData() const = 0;

};
