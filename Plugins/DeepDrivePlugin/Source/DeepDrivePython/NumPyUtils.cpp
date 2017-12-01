
#include "NumPyUtils.h"

#include "Public/DeepDriveDataTypes.h"
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include <iostream>

void NumPyUtils::copyVector3(const DeepDriveVector3 &src, PyArrayObject *dst)
{
	PyArray_SETITEM(dst, PyArray_GETPTR1(dst, 0), PyFloat_FromDouble(src.x));
	PyArray_SETITEM(dst, PyArray_GETPTR1(dst, 1), PyFloat_FromDouble(src.y));
	PyArray_SETITEM(dst, PyArray_GETPTR1(dst, 2), PyFloat_FromDouble(src.z));
}


bool NumPyUtils::getVector3(PyObject *src, float dst[3], bool isPyArray)
{
	bool res = false;
	if(isPyArray && PyArray_NDIM(src) == 1 && PyArray_DIM(src, 0) == 3)
	{
		res = true;
		for(unsigned i = 0; res && i < 3; ++i)
		{
			PyObject *item = PyArray_GETITEM(src, PyArray_GETPTR1(src, i));
			if(PyFloat_Check(item))
			{
				dst[i] = static_cast<float> (PyFloat_AsDouble(item));
			}
			else if(PyLong_Check(item))
			{
				dst[i] = static_cast<float> (PyLong_AsLong(item));
			}
			else
			{
				res = false;
			}
		}
	}
	else if(PyList_Check(src) && PyList_Size(src) >= 3)
	{
		res = true;
		for(unsigned i = 0; res && i < 3; ++i)
		{
			PyObject *item = PyList_GetItem(src, i);
			if(PyFloat_Check(item))
			{
				dst[i] = static_cast<float> (PyFloat_AsDouble(item));
			}
			else if(PyLong_Check(item))
			{
				dst[i] = static_cast<float> (PyLong_AsLong(item));
			}
			else
			{
				res = false;
			}
		}
	}
	return res;
}
