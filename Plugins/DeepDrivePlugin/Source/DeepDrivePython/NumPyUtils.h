
#pragma once

#include "numpy/arrayobject.h"

struct DeepDriveVector3;

class NumPyUtils
{
public:

	static void copyVector3(const DeepDriveVector3 &src, PyArrayObject *dst);

};
