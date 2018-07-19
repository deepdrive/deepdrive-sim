
#pragma once

#include "Python.h"
#include "structmember.h"

#include "common/NumPyUtils.h"

#include <iostream>

/*	Forward declarations
*/
static PyObject* PySimulationConfigurationObject_new_impl();

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

/*
- Screen Type
- Resolution
- Resolution Scale
- VSync
- Texture quality
- Shadow quality
- Effects quality
- Post process level
- Motion Blur
- View distance
- Ambient Occlusion
*/

struct PySimulationConfigurationObject
{
	PyObject_HEAD

	uint32			is_fullscreen;

	uint32			vsync_enabled;

	uint32			resolution_width;

	uint32			resolution_height;

	double			resolution_scale;

	uint32			texture_quality;

	uint32			shadow_quality;

	uint32			effect_quality;

	uint32			post_process_level;

	uint32			motion_blur_quality;

	uint32			view_distance;

	uint32			ambient_occlusion;

};

static PyMemberDef PySimulationConfigurationMembers[] =
{
	{"is_fullscreen", T_UINT, offsetof(PySimulationConfigurationObject, is_fullscreen), 0, "Fullscrfeen or windowed"},
	{"vsync_enabled", T_UINT, offsetof(PySimulationConfigurationObject, vsync_enabled), 0, "VSync enabled"},
	{"resolution_width", T_UINT, offsetof(PySimulationConfigurationObject, resolution_width), 0, "Resolution width"},
	{"resolution_height", T_UINT, offsetof(PySimulationConfigurationObject, resolution_height), 0, "Resolution height"},
	{"resolution_scale", T_DOUBLE, offsetof(PySimulationConfigurationObject, resolution_scale), 0, "Resolution scale"},
	{"texture_quality", T_UINT, offsetof(PySimulationConfigurationObject, texture_quality), 0, "Texture quality level (0 - 3)"},
	{"shadow_quality", T_UINT, offsetof(PySimulationConfigurationObject, shadow_quality), 0, "Shadow quality level (0 - 3)"},
	{"effect_quality", T_UINT, offsetof(PySimulationConfigurationObject, effect_quality), 0, "Effect quality level (0 - 3)"},
	{"post_process_level", T_UINT, offsetof(PySimulationConfigurationObject, post_process_level), 0, "Post process level (0 - 3)"},
	{"motion_blur_quality", T_UINT, offsetof(PySimulationConfigurationObject, motion_blur_quality), 0, "Motion blur quality level (0 - 3)"},
	{"view_distance", T_UINT, offsetof(PySimulationConfigurationObject, view_distance), 0, "View distance level (0 - 3)"},
	{"ambient_occlusion", T_UINT, offsetof(PySimulationConfigurationObject, ambient_occlusion), 0, "Ambient occlusion quality level (0 - 3)"},
	{NULL}
};

static PyObject* PySimulationConfigurationObject_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	return PySimulationConfigurationObject_new_impl();
}

static void PySimulationConfigurationObject_init_impl(PySimulationConfigurationObject *self)
{
//	std::cout << "PySimulationConfigurationObject_init_impl\n";
	self->is_fullscreen = 0;
	self->vsync_enabled = 0;
	self->resolution_width = 1920;
	self->resolution_height = 1080;
	self->resolution_scale = 1.0;
	self->texture_quality = 3;
	self->shadow_quality = 3;
	self->effect_quality = 3;
	self->post_process_level = 3;
	self->motion_blur_quality = 3;
	self->view_distance = 3;
	self->ambient_occlusion = 3;
}

static int PySimulationConfigurationObject_init(PyObject *self, PyObject *args, PyObject *kwds)
{
	std::cout << "PySimulationConfigurationObject_init\n";

	PySimulationConfigurationObject_init_impl(reinterpret_cast<PySimulationConfigurationObject*> (self));

	return 0;
}

static PyTypeObject PySimulationConfigurationType =
{
	PyVarObject_HEAD_INIT(NULL, 0)
	"SimulationConfiguration",		//	tp name
	sizeof(PySimulationConfigurationObject), 		//	tp_basicsize
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
	"Simulation configuration",		//tp_doc
	0,		//	tp_traverse
	0,		//	tp_clear
	0,		//	tp_richcompare
	0,		//	tp_weaklistoffset
	0,		//	tp_iter
	0,		//	tp_iternext
	0,		//	tp_methods
	PySimulationConfigurationMembers,		//	tp_members
	0,		//	tp_getset
	0,		//	tp_base
	0,		//	tp_dict
	0,		//	tp_descr_get
	0,		//	tp_descr_set
	0,		//	tp_dictoffset
	PySimulationConfigurationObject_init,		//	tp_init
	0,		//	tp_alloc
	PySimulationConfigurationObject_new,		//	tp_new
};


static PyObject* PySimulationConfigurationObject_new_impl()
{
	PySimulationConfigurationObject *self;

	if (PyType_Ready(&PySimulationConfigurationType) < 0)
		return NULL;

	self = PyObject_New(PySimulationConfigurationObject, &PySimulationConfigurationType);

	if(self)
	{
		PySimulationConfigurationObject_init_impl(self);
	}

	return (PyObject *)self;
}
