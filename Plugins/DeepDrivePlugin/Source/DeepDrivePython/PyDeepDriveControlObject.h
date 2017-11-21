
#pragma once

#include "Python.h"
#include "structmember.h"

#include "NumPyUtils.h"

#include <iostream>

/*	Forward declarations
*/
static PyObject* PyDeepDriveControlObject_new_impl();


struct PyDeepDriveControlObject
{
	PyObject_HEAD

	double						capture_timestamp;

	uint32						capture_sequence_number;

	uint32						should_reset;

	double						steering;

	double						throttle;

	double						brake;

	double						handbrake;

	uint32						is_game_driving;
};

static PyMemberDef PyDeepDriveControlMembers[] =
{
	{"steering", T_DOUBLE, offsetof(PyDeepDriveControlObject, steering), 0, "Steering in range [-1, 1]"}
,	{"throttle", T_DOUBLE, offsetof(PyDeepDriveControlObject, throttle), 0, "Throttle in range [-1, 1]"}
,	{"brake", T_DOUBLE, offsetof(PyDeepDriveControlObject, brake), 0, "Brake in range [-1, 1]"}
,	{"handbrake", T_DOUBLE, offsetof(PyDeepDriveControlObject, handbrake), 0, "Handbrake"}
,	{"is_game_driving", T_UINT, offsetof(PyDeepDriveControlObject, is_game_driving), 0, "Should the game drive"}
,	{"should_reset", T_UINT, offsetof(PyDeepDriveControlObject, should_reset), 0, "Should the car respawn"}
,	{"capture_timestamp", T_DOUBLE, offsetof(PyDeepDriveControlObject, capture_timestamp), 0, "Timestamp of capture this control message is based on"}
,	{"capture_sequence_number", T_UINT, offsetof(PyDeepDriveControlObject, capture_sequence_number), 0, "Sequence number of capture this control message is based on"}
,	{NULL}
};

static PyObject* PyDeepDriveControlObject_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	return PyDeepDriveControlObject_new_impl();
}

static void PyDeepDriveControlObject_init_impl(PyDeepDriveControlObject *self)
{
//	std::cout << "PyDeepDriveControlObject_init_impl\n";
	if(self)
	{
		self->steering = 0.0;
		self->throttle = 0.0;
		self->brake = 0.0;
		self->handbrake = 0;
	}
}

static int PyDeepDriveControlObject_init(PyObject *self, PyObject *args, PyObject *kwds)
{
//	std::cout << "PyDeepDriveControlObject_init\n";

	PyDeepDriveControlObject_init_impl(reinterpret_cast<PyDeepDriveControlObject*> (self));

	return 0;
}

static PyTypeObject PyDeepDriveControlType =
{
	PyVarObject_HEAD_INIT(NULL, 0)
	"DeepDriveControl",		//	tp name
	sizeof(PyDeepDriveControlObject), 		//	tp_basicsize
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
	"DeepDrive control",		//tp_doc
	0,		//	tp_traverse
	0,		//	tp_clear
	0,		//	tp_richcompare
	0,		//	tp_weaklistoffset
	0,		//	tp_iter
	0,		//	tp_iternext
	0,		//	tp_methods
	PyDeepDriveControlMembers,		//	tp_members
	0,		//	tp_getset
	0,		//	tp_base
	0,		//	tp_dict
	0,		//	tp_descr_get
	0,		//	tp_descr_set
	0,		//	tp_dictoffset
	PyDeepDriveControlObject_init,		//	tp_init
	0,		//	tp_alloc
	PyDeepDriveControlObject_new,		//	tp_new
};


static PyObject* PyDeepDriveControlObject_new_impl()
{
	PyDeepDriveControlObject *self;

	if (PyType_Ready(&PyDeepDriveControlType) < 0)
		return NULL;

	self = PyObject_New(PyDeepDriveControlObject, &PyDeepDriveControlType);

	if(self)
	{
		PyDeepDriveControlObject_init_impl(self);
	}

	return (PyObject *)self;
}
