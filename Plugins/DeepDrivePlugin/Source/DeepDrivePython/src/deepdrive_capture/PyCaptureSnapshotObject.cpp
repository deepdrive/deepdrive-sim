
#include "PyCaptureSnapshotObject.h"
#include "PyCaptureLastCollisionObject.h"

#include <iostream>

int PyCaptureSnapshotObject::initNumPy()
{
    std::cout << "Init numPy for PyCaptureSnapshotObject\n";
    import_array();

	return 0;
}


PyArrayObject* PyCaptureSnapshotObject::createVec3()
{
	npy_intp dims[1] = {3};
	PyArrayObject *vec3 = reinterpret_cast<PyArrayObject*> (PyArray_SimpleNew(1, dims, NPY_DOUBLE));

	return vec3;
}

void PyCaptureSnapshotObject::init(PyCaptureSnapshotObject *self)
{
	// std::cout << "PyCaptureSnapshotObject_init_impl\n";
	self->position = createVec3();
	self->rotation = createVec3();
	self->velocity = createVec3();
	self->acceleration = createVec3();
	self->angular_velocity = createVec3();
	self->angular_acceleration = createVec3();
	self->forward_vector = createVec3();
	self->up_vector = createVec3();
	self->right_vector = createVec3();
	self->dimension = createVec3();

	self->last_collision = reinterpret_cast<PyCaptureLastCollisionObject*> (PyCaptureLastCollisionType.tp_new(&PyCaptureLastCollisionType, 0, 0));
}


PyObject* PyCaptureSnapshotObject::allocate()
{
	PyCaptureSnapshotObject *self;

	if (PyType_Ready(&PyCaptureSnapshotType) < 0)
		return NULL;

	self = PyObject_New(PyCaptureSnapshotObject, &PyCaptureSnapshotType);

	if(self)
	{
		PyCaptureSnapshotObject::init(self);
	}

	return (PyObject *)self;
}
