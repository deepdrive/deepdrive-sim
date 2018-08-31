

#include "PyCaptureLastCollisionObject.h"

#include <iostream>

int PyCaptureLastCollisionObject::initNumPy()
{
	std::cout << "Init numPy for PyCaptureLastCollisionObject\n";
	import_array();
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
	int dims[1] = {3};

	self->time_utc = 0;
	self->time_stamp = 0.0;
	self->time_since_last_collision = 0.0;

	self->collision_location[0] = 0;

	self->collidee_velocity = reinterpret_cast<PyArrayObject*> (PyArray_FromDims(1, dims, NPY_DOUBLE));
	self->collision_normal = reinterpret_cast<PyArrayObject*> (PyArray_FromDims(1, dims, NPY_DOUBLE));
}
