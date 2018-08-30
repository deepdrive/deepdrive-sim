#include "PyCaptureCameraObject.h"
#include <iostream>

int PyCaptureCameraObject::initNumPy()
{
	std::cout << "Init numPy for PyCaptureCameraObject\n";
	import_array();
	return 0;
}

void PyCaptureCameraObject::init(PyCaptureCameraObject *self)
{
}

PyObject* PyCaptureCameraObject::allocate()
{
	PyCaptureCameraObject *self;

	if (PyType_Ready(&PyCaptureCameraType) < 0)
		return NULL;

	self = PyObject_New(PyCaptureCameraObject, &PyCaptureCameraType);

//	if(self)
//	{
//		PyCaptureCameraObject::init(self);
//	}

	return (PyObject *)self;
}

PyArrayObject* PyCaptureCameraObject::createImage(int size, const uint8_t *data)
{
	npy_intp dims[1] = {size};
	return reinterpret_cast<PyArrayObject*> (PyArray_SimpleNewFromData(1, dims, NPY_FLOAT16, const_cast<uint8_t*> (data)));
}
