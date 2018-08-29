
#pragma once

#include "Python.h"
#include "structmember.h"

#include "common/NumPyUtils.h"

#include <iostream>

struct PyCaptureLastCollisionObject
{
	PyObject_HEAD

	int64_t				time_utc;

	double				time_stamp;

	double				time_since_last_collision;

	//char				collision_location[DeepDriveMessageHeader::StringSize];

	PyArrayObject		*collidee_velocity;

	PyArrayObject		*collision_normal;

	static int initNumPy();
	static void init(PyCaptureLastCollisionObject *self);
	static PyObject* allocate();

};

static PyMemberDef PyCaptureLastCollisionMembers[] =
{
	{"time_utc", T_LONG, offsetof(PyCaptureLastCollisionObject, time_utc), 0, "Capture snapshot sequence number"}
,	{"time_stamp", T_DOUBLE, offsetof(PyCaptureLastCollisionObject, time_stamp), 0, "Timestamp of last collision"}
,	{"time_since_last_collision", T_DOUBLE, offsetof(PyCaptureLastCollisionObject, time_since_last_collision), 0, "Times since last collision of last collision"}
,	{"collidee_velocity", T_OBJECT_EX, offsetof(PyCaptureLastCollisionObject, collidee_velocity), 0, "x,y,z velocity of vehicle in frame of ego origin orientation, cm/s"}
,	{"collision_normal", T_OBJECT_EX, offsetof(PyCaptureLastCollisionObject, collision_normal), 0, "x,y,z velocity of vehicle in frame of ego origin orientation, cm/s"}
,	{NULL}
};

static PyObject* PyCaptureLastCollisionObject_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	return PyCaptureLastCollisionObject::allocate();
}

static int PyCaptureLastCollisionObject_init(PyObject *self, PyObject *args, PyObject *kwds)
{
	std::cout << "PyCaptureLastCollisionObject_init\n";

	PyCaptureLastCollisionObject::init(reinterpret_cast<PyCaptureLastCollisionObject*> (self));

	return 0;
}

static PyTypeObject PyCaptureLastCollisionType =
{
	PyVarObject_HEAD_INIT(NULL, 0)
	"CaptureLastCollision",		//	tp name
	sizeof(PyCaptureLastCollisionObject), 		//	tp_basicsize
	0,		//	tp_itemsize
	0,		//	tp_dealloc
	0,		//	tp_print
	0,		//	tp_getattr
	0,		//	tp_setattr
	0,		//	tp_reserved
	0,		//	tp_repr
	0,		//	tp_as_number
	0,		//	tp_as_sequence
	0,		//	tp_as_mapping
	0,		//	tp_hash
	0,		//	tp_call
	0,		//	tp_str
	0,		//	tp_getattro
	0,		//	tp_setattro
	0,		//	tp_as_buffer
	Py_TPFLAGS_DEFAULT,		//	tp_flags
	"Capture LastCollision",		//tp_doc
	0,		//	tp_traverse
	0,		//	tp_clear
	0,		//	tp_richcompare
	0,		//	tp_weaklistoffset
	0,		//	tp_iter
	0,		//	tp_iternext
	0,		//	tp_methods
	PyCaptureLastCollisionMembers,		//	tp_members
	0,		//	tp_getset
	0,		//	tp_base
	0,		//	tp_dict
	0,		//	tp_descr_get
	0,		//	tp_descr_set
	0,		//	tp_dictoffset
	PyCaptureLastCollisionObject_init,		//	tp_init
	0,		//	tp_alloc
	PyCaptureLastCollisionObject_new,		//	tp_new
};
