
#include "PyMultiAgentSnapshotObject.h"
// #include "PyCaptureLastCollisionObject.h"

#include <iostream>

int PyMultiAgentSnapshotObject::initNumPy()
{
    std::cout << "Init numPy for PyMultiAgentSnapshotObject\n";
    import_array();

	return 0;
}


PyArrayObject* PyMultiAgentSnapshotObject::createVec3()
{
	npy_intp dims[1] = {3};
	PyArrayObject *vec3 = reinterpret_cast<PyArrayObject*> (PyArray_SimpleNew(1, dims, NPY_DOUBLE));

	return vec3;
}

void PyMultiAgentSnapshotObject::init(PyMultiAgentSnapshotObject *self)
{
	// std::cout << "PyMultiAgentSnapshotObject_init_impl\n";
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

	// self->last_collision = reinterpret_cast<PyCaptureLastCollisionObject*> (PyCaptureLastCollisionType.tp_new(&PyCaptureLastCollisionType, 0, 0));
}


PyObject* PyMultiAgentSnapshotObject::allocate()
{
	PyMultiAgentSnapshotObject *self;

	if (PyType_Ready(&PyMultiAgentSnapshotType) < 0)
		return NULL;

	self = PyObject_New(PyMultiAgentSnapshotObject, &PyMultiAgentSnapshotType);

	if(self)
	{
		PyMultiAgentSnapshotObject::init(self);
	}

	return (PyObject *)self;
}
