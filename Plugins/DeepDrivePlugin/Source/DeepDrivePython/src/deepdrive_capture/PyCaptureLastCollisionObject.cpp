

#include "PyCaptureLastCollisionObject.h"

#include <iostream>

int PyCaptureLastCollisionObject::initNumPy()
{
	static bool initRequired = true;
	if(initRequired)
	{
		std::cout << "Init numPy for PyCaptureLastCollisionObject\n";
		import_array();
		initRequired = false;
	}
	return 0;
}

PyObject* PyCaptureLastCollisionObject::allocate()
{
	PyCaptureLastCollisionObject *self;

	if (PyType_Ready(&PyCaptureLastCollisionType) < 0)
		return NULL;

	self = PyObject_New(PyCaptureLastCollisionObject, &PyCaptureLastCollisionType);

	if(self)
	{
		PyCaptureLastCollisionObject::init(self);
	}

	return (PyObject *)self;
}

void PyCaptureLastCollisionObject::init(PyCaptureLastCollisionObject *self)
{
	(void) PyCaptureLastCollisionObject::initNumPy();
	int dims[1] = {3};

	self->collidee_velocity = reinterpret_cast<PyArrayObject*> (PyArray_FromDims(1, dims, NPY_DOUBLE));
	self->collision_normal = reinterpret_cast<PyArrayObject*> (PyArray_FromDims(1, dims, NPY_DOUBLE));
}
