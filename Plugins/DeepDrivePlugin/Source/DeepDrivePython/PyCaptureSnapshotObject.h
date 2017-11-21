
#pragma once

#include "Python.h"
#include "structmember.h"

#include "NumPyUtils.h"

#include <iostream>

/*	Forward declarations
*/
static PyObject* PyCaptureSnapshotObject_new_impl();

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


struct PyCaptureSnapshotObject
{
	PyObject_HEAD

	double				capture_timestamp;

	uint32				sequence_number;

	uint32				is_game_driving;

	uint32				is_resetting;

	double				speed;

	double				steering;

	double				throttle;

	double				brake;

	uint32				handbrake;

	PyArrayObject*		position;

	PyArrayObject*		rotation;

	PyArrayObject*		velocity;

	PyArrayObject*		acceleration;

	PyArrayObject*		angular_velocity;

	PyArrayObject*		angular_acceleration;

	PyArrayObject*		forward_vector;

	PyArrayObject*		up_vector;

	PyArrayObject*		right_vector;

	PyArrayObject*		dimension;

	double				distance_along_route;

	double				distance_to_center_of_lane;

	uint32				lap_number;

	uint32				camera_count;

	PyListObject		*cameras;

};

static PyMemberDef PyCaptureSnapshotMembers[] =
{
	{"capture_timestamp", T_DOUBLE, offsetof(PyCaptureSnapshotObject, capture_timestamp), 0, "Timestamp the capture was taken"},
	{"sequence_number", T_UINT, offsetof(PyCaptureSnapshotObject, sequence_number), 0, "Capture snapshot sequence number"},
	{"is_game_driving", T_UINT, offsetof(PyCaptureSnapshotObject, is_game_driving), 0, "Is game driving"},
	{"is_resetting", T_UINT, offsetof(PyCaptureSnapshotObject, is_game_driving), 0, "Is the car respawning"},
	{"speed", T_DOUBLE, offsetof(PyCaptureSnapshotObject, speed), 0, "speed"},
	{"steering", T_DOUBLE, offsetof(PyCaptureSnapshotObject, steering), 0, "0 is straight ahead, -1 is left, 1 is right"},
	{"throttle", T_DOUBLE, offsetof(PyCaptureSnapshotObject, throttle), 0, "0 is coast / idle, -1 is reverse, 1 is full throttle ahead"},
	{"brake", T_DOUBLE, offsetof(PyCaptureSnapshotObject, brake), 0, "0 to 1, 0 is no brake, 1 is full brake"},
	{"handbrake", T_UINT, offsetof(PyCaptureSnapshotObject, handbrake), 0, "Handbrake on or off"},
	{"position", T_OBJECT_EX, offsetof(PyCaptureSnapshotObject, position), 0, "x,y,z of wheel base center??? from origin coordinates of ego vehicle, cm"},
	{"rotation", T_OBJECT_EX, offsetof(PyCaptureSnapshotObject, rotation), 0, "roll, pitch, yaw of vehicle, in degrees, UE4 style"},
	{"velocity", T_OBJECT_EX, offsetof(PyCaptureSnapshotObject, velocity), 0, "x,y,z velocity of vehicle in frame of ego origin orientation, cm/s"},
	{"acceleration", T_OBJECT_EX, offsetof(PyCaptureSnapshotObject, acceleration), 0, "Current acceleration"},
	{"angular_velocity", T_OBJECT_EX, offsetof(PyCaptureSnapshotObject, angular_velocity), 0, "Current angular velocity"},
	{"angular_acceleration", T_OBJECT_EX, offsetof(PyCaptureSnapshotObject, angular_acceleration), 0, "Current angular acceleration"},
	{"forward_vector", T_OBJECT_EX, offsetof(PyCaptureSnapshotObject, forward_vector), 0, "Current forward vector"},
	{"up_vector", T_OBJECT_EX, offsetof(PyCaptureSnapshotObject, up_vector), 0, "Current up vector"},
	{"right_vector", T_OBJECT_EX, offsetof(PyCaptureSnapshotObject, right_vector), 0, "Current right vector"},
	{"dimension", T_OBJECT_EX, offsetof(PyCaptureSnapshotObject, dimension), 0, "Current dimension"},
	{"distance_along_route", T_DOUBLE, offsetof(PyCaptureSnapshotObject, distance_along_route), 0, "Distance achieved to destination on designated route in cm"},
	{"distance_to_center_of_lane", T_DOUBLE, offsetof(PyCaptureSnapshotObject, distance_to_center_of_lane), 0, "Last distance to previously achieved waypoint - where waypoints are 4m apart"},
	{"lap_number", T_UINT, offsetof(PyCaptureSnapshotObject, lap_number), 0, "Number of laps achieved since last reset"},
	{"camera_count", T_UINT, offsetof(PyCaptureSnapshotObject, camera_count), 0, "Number of captured cameras"},
	{"cameras", T_OBJECT_EX, offsetof(PyCaptureSnapshotObject, cameras), 0, "List of captured cameras"},
	{NULL}
};

static PyObject* PyCaptureSnapshotObject_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	return PyCaptureSnapshotObject_new_impl();
}

static PyArrayObject* createVec3()
{
	(void) initNumPy();

	int dims[1] = {3};
	PyArrayObject *vec3 = reinterpret_cast<PyArrayObject*> (PyArray_FromDims(1, dims, NPY_DOUBLE));

	return vec3;
}

static void PyCaptureSnapshotObject_init_impl(PyCaptureSnapshotObject *self)
{
//	std::cout << "PyCaptureSnapshotObject_init_impl\n";
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
}

static int PyCaptureSnapshotObject_init(PyObject *self, PyObject *args, PyObject *kwds)
{
	std::cout << "PyCaptureSnapshotObject_init\n";

	PyCaptureSnapshotObject_init_impl(reinterpret_cast<PyCaptureSnapshotObject*> (self));

	return 0;
}

static PyTypeObject PyCaptureSnapshotType =
{
	PyVarObject_HEAD_INIT(NULL, 0)
	"CaptureSnapshot",		//	tp name
	sizeof(PyCaptureSnapshotObject), 		//	tp_basicsize
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
	"Capture snapshot",		//tp_doc
	0,		//	tp_traverse
	0,		//	tp_clear
	0,		//	tp_richcompare
	0,		//	tp_weaklistoffset
	0,		//	tp_iter
	0,		//	tp_iternext
	0,		//	tp_methods
	PyCaptureSnapshotMembers,		//	tp_members
	0,		//	tp_getset
	0,		//	tp_base
	0,		//	tp_dict
	0,		//	tp_descr_get
	0,		//	tp_descr_set
	0,		//	tp_dictoffset
	PyCaptureSnapshotObject_init,		//	tp_init
	0,		//	tp_alloc
	PyCaptureSnapshotObject_new,		//	tp_new
};


static PyObject* PyCaptureSnapshotObject_new_impl()
{
	PyCaptureSnapshotObject *self;

	if (PyType_Ready(&PyCaptureSnapshotType) < 0)
		return NULL;

	self = PyObject_New(PyCaptureSnapshotObject, &PyCaptureSnapshotType);

	if(self)
	{
		PyCaptureSnapshotObject_init_impl(self);
	}

	return (PyObject *)self;
}
