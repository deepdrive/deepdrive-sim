
#pragma once

#include "Python.h"
#include "structmember.h"

#include "common/NumPyUtils.h"

#include <stdint.h>

// struct PyCaptureLastCollisionObject;

struct PyMultiAgentSnapshotObject
{
	static int initNumPy();
	static void init(PyMultiAgentSnapshotObject *self);
	static PyObject* allocate();
	static PyArrayObject* createVec3();


	PyObject_HEAD

	uint32_t			agent_id;

	double				snapshot_timestamp;

	double				speed;

	double				steering;

	double				throttle;

	double				brake;

	uint32_t			handbrake;

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

	double				route_length;

	double				distance_to_center_of_lane;

	// PyCaptureLastCollisionObject	*last_collision;

};

static PyMemberDef PyMultiAgentSnapshotMembers[] =
{
	{"agent_id", T_UINT, offsetof(PyMultiAgentSnapshotObject, agent_id), 0, "Id of agent"},
	{"snapshot_timestamp", T_DOUBLE, offsetof(PyMultiAgentSnapshotObject, snapshot_timestamp), 0, "Timestamp the capture was taken"},
	{"speed", T_DOUBLE, offsetof(PyMultiAgentSnapshotObject, speed), 0, "speed"},
	{"steering", T_DOUBLE, offsetof(PyMultiAgentSnapshotObject, steering), 0, "0 is straight ahead, -1 is left, 1 is right"},
	{"throttle", T_DOUBLE, offsetof(PyMultiAgentSnapshotObject, throttle), 0, "0 is coast / idle, -1 is reverse, 1 is full throttle ahead"},
	{"brake", T_DOUBLE, offsetof(PyMultiAgentSnapshotObject, brake), 0, "0 to 1, 0 is no brake, 1 is full brake"},
	{"handbrake", T_UINT, offsetof(PyMultiAgentSnapshotObject, handbrake), 0, "Handbrake on or off"},
	{"position", T_OBJECT_EX, offsetof(PyMultiAgentSnapshotObject, position), 0, "x,y,z of wheel base center??? from origin coordinates of ego vehicle, cm"},
	{"rotation", T_OBJECT_EX, offsetof(PyMultiAgentSnapshotObject, rotation), 0, "roll, pitch, yaw of vehicle, in degrees, UE4 style"},
	{"velocity", T_OBJECT_EX, offsetof(PyMultiAgentSnapshotObject, velocity), 0, "x,y,z velocity of vehicle in frame of ego origin orientation, cm/s"},
	{"acceleration", T_OBJECT_EX, offsetof(PyMultiAgentSnapshotObject, acceleration), 0, "Current acceleration"},
	{"angular_velocity", T_OBJECT_EX, offsetof(PyMultiAgentSnapshotObject, angular_velocity), 0, "Current angular velocity"},
	{"angular_acceleration", T_OBJECT_EX, offsetof(PyMultiAgentSnapshotObject, angular_acceleration), 0, "Current angular acceleration"},
	{"forward_vector", T_OBJECT_EX, offsetof(PyMultiAgentSnapshotObject, forward_vector), 0, "Current forward vector"},
	{"up_vector", T_OBJECT_EX, offsetof(PyMultiAgentSnapshotObject, up_vector), 0, "Current up vector"},
	{"right_vector", T_OBJECT_EX, offsetof(PyMultiAgentSnapshotObject, right_vector), 0, "Current right vector"},
	{"dimension", T_OBJECT_EX, offsetof(PyMultiAgentSnapshotObject, dimension), 0, "Current dimension"},
	{"distance_along_route", T_DOUBLE, offsetof(PyMultiAgentSnapshotObject, distance_along_route), 0, "Distance achieved to destination on designated route in cm"},
	{"route_length", T_DOUBLE, offsetof(PyMultiAgentSnapshotObject, route_length), 0, "Total length of current route."},
	{"distance_to_center_of_lane", T_DOUBLE, offsetof(PyMultiAgentSnapshotObject, distance_to_center_of_lane), 0, "Last distance to previously achieved waypoint - where waypoints are 4m apart"},
	// {"last_collision", T_OBJECT_EX, offsetof(PyMultiAgentSnapshotObject, last_collision), 0, "Last collision data"},
	{NULL}
};

static PyObject* PyMultiAgentSnapshotObject_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	return PyMultiAgentSnapshotObject::allocate();
}

static int PyMultiAgentSnapshotObject_init(PyObject *self, PyObject *args, PyObject *kwds)
{
	PyMultiAgentSnapshotObject::init(reinterpret_cast<PyMultiAgentSnapshotObject*> (self));
	return 0;
}

static PyTypeObject PyMultiAgentSnapshotType =
{
	PyVarObject_HEAD_INIT(NULL, 0)
	"AgentSnapshot",		//	tp name
	sizeof(PyMultiAgentSnapshotObject), 		//	tp_basicsize
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
	PyMultiAgentSnapshotMembers,		//	tp_members
	0,		//	tp_getset
	0,		//	tp_base
	0,		//	tp_dict
	0,		//	tp_descr_get
	0,		//	tp_descr_set
	0,		//	tp_dictoffset
	PyMultiAgentSnapshotObject_init,		//	tp_init
	0,		//	tp_alloc
	PyMultiAgentSnapshotObject_new,		//	tp_new
};
