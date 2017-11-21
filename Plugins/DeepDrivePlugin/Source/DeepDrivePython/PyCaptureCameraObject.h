
#pragma once

#include "Python.h"
#include "structmember.h"

#include "NumPyUtils.h"

#include <iostream>

/*	Forward declarations
*/
static PyObject* PyCaptureCameraObject_new_impl();

#if 0
static int initNumPy()
{
	static bool initRequired = true;
	if(initRequired)
	{
//		std::cout << "Init numPy\n";
		import_array();
		initRequired = false;
	}
	return 0;
}
#endif

struct PyCaptureCameraObject
{
	PyObject_HEAD

	uint32				type;
	uint32				id;

	double				horizontal_field_of_view;
	double				aspect_ratio;


	uint32				capture_width;
	uint32				capture_height;

	PyArrayObject		*image_data;
	PyArrayObject		*depth_data;

};

static PyMemberDef PyCaptureCameraMembers[] =
{
	{"type", T_UINT, offsetof(PyCaptureCameraObject, type), 0, "Capture snapshot sequence number"}
,	{"id", T_UINT, offsetof(PyCaptureCameraObject, id), 0, "Capture snapshot sequence number"}
,	{"horizontal_field_of_view", T_DOUBLE, offsetof(PyCaptureCameraObject, horizontal_field_of_view), 0, "Horizontal field of view in radians"}
,	{"aspect_ratio", T_DOUBLE, offsetof(PyCaptureCameraObject, aspect_ratio), 0, "Aspect ratio"}
,	{"capture_width", T_UINT, offsetof(PyCaptureCameraObject, capture_width), 0, "Capture width"}
,	{"capture_height", T_UINT, offsetof(PyCaptureCameraObject, capture_height), 0, "Capture height"}
,	{"image_data", T_OBJECT_EX, offsetof(PyCaptureCameraObject, image_data), 0, "Image data"}
,	{"depth_data", T_OBJECT_EX, offsetof(PyCaptureCameraObject, depth_data), 0, "Depth data"}
,	{NULL}
};

static PyObject* PyCaptureCameraObject_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	return PyCaptureCameraObject_new_impl();
}

static void PyCaptureCameraObject_init_impl(PyCaptureCameraObject *self)
{
//	(void) initNumPy();

//	std::cout << "PyCaptureCameraObject_init_impl\n";
}

static int PyCaptureCameraObject_init(PyObject *self, PyObject *args, PyObject *kwds)
{
	std::cout << "PyCaptureCameraObject_init\n";

	PyCaptureCameraObject_init_impl(reinterpret_cast<PyCaptureCameraObject*> (self));

	return 0;
}

static PyTypeObject PyCaptureCameraType =
{
	PyVarObject_HEAD_INIT(NULL, 0)
	"CaptureCamera",		//	tp name
	sizeof(PyCaptureCameraObject), 		//	tp_basicsize
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
	"Capture camera",		//tp_doc
	0,		//	tp_traverse
	0,		//	tp_clear
	0,		//	tp_richcompare
	0,		//	tp_weaklistoffset
	0,		//	tp_iter
	0,		//	tp_iternext
	0,		//	tp_methods
	PyCaptureCameraMembers,		//	tp_members
	0,		//	tp_getset
	0,		//	tp_base
	0,		//	tp_dict
	0,		//	tp_descr_get
	0,		//	tp_descr_set
	0,		//	tp_dictoffset
	PyCaptureCameraObject_init,		//	tp_init
	0,		//	tp_alloc
	PyCaptureCameraObject_new,		//	tp_new
};


static PyObject* PyCaptureCameraObject_new_impl()
{
	PyCaptureCameraObject *self;

	if (PyType_Ready(&PyCaptureCameraType) < 0)
		return NULL;

	self = PyObject_New(PyCaptureCameraObject, &PyCaptureCameraType);

	if(self)
	{
		PyCaptureCameraObject_init_impl(self);
	}

	return (PyObject *)self;
}
