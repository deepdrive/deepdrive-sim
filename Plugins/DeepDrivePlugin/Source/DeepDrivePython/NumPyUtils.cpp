
#include "NumPyUtils.h"

#include "Public/DeepDriveDataTypes.h"

#include <iostream>

void NumPyUtils::copyVector3(const DeepDriveVector3 &src, PyArrayObject *dst)
{
	PyArray_SETITEM(dst, PyArray_GETPTR1(dst, 0), PyFloat_FromDouble(src.x));
	PyArray_SETITEM(dst, PyArray_GETPTR1(dst, 1), PyFloat_FromDouble(src.y));
	PyArray_SETITEM(dst, PyArray_GETPTR1(dst, 2), PyFloat_FromDouble(src.z));
}
