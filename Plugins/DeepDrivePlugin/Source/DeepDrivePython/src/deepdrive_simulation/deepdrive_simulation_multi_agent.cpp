
#include "Python.h"
#include "Engine.h"

#include "deepdrive_simulation/DeepDriveSimulation.hpp"
#include "deepdrive_simulation/deepdrive_simulation_error.h"

#include "deepdrive_simulation/PyMultiAgentSnapshotObject.h"

#include <iostream>

/*	Retrieve lists of available agents
 *
 *
 *	@return	List of available list
*/
static PyObject *multi_agent_get_agents_list(PyObject *self, PyObject *args)
{
	PyObject *ret = PyList_New(0);
	if (DeepDriveSimulation::getInstance())
	{
		std::vector<uint32> list;
		const int32 requestRes = DeepDriveSimulation::getInstance()->getAgentsList(list);
		if (requestRes >= 0)
		{
			for(auto id : list)
				PyList_Append(ret, PyLong_FromUnsignedLong(id));
		}
		else
			return handleError(requestRes);
	}
	else
		std::cout << "Not connect to Simulation server\n";

	return ret;
}

/*	Request remote control for a set of specified agent
 *
 *	@param	[uint32]	List of agent id's to be remotely controlled. If list is empty or None, all agents will be remotely controlled.
 *	@return	True, if successfully, otherwise false
*/
static PyObject *multi_agent_request_control(PyObject *self, PyObject *args)
{
	uint32 res = 0;

	if(DeepDriveSimulation::getInstance())
	{
		PyObject *srcList = 0;
		int32 ok = PyArg_ParseTuple(args, "O!", &PyList_Type, &srcList);
		if (ok)
		{
			std::vector<uint32> agentIds;
			int32 n = PyList_Size(srcList);
			for(int32 i = 0; i < n; ++i)
			{
				PyObject *pItem = PyList_GetItem(srcList, i);
				if (!PyLong_Check(pItem))
				{
					PyErr_SetString(PyExc_TypeError, "List items must be integers");
					return NULL;
				}
				const uint32 id = PyLong_AsLong(pItem);
				agentIds.push_back(id);
			}
			const int32 initRes = DeepDriveSimulation::getInstance()->requestControl(agentIds);
			if(initRes >= 0)
				res = static_cast<uint32> (initRes);
			else
				return handleError(initRes);
		}
		else
			std::cout << "Wrong arguments\n";
	}
	else
		std::cout << "Not connect to Simulation server\n";

	return Py_BuildValue("i", res);
}

/*	Release remote control for a set of specified agent
 *
 *	@param	[uint32]	List of agent id's to be remotely controlled. If list is empty or None, all agents will be remotely controlled.
 *	@return	True, if successfully, otherwise false
*/
static PyObject *multi_agent_release_control(PyObject *self, PyObject *args)
{
	uint32 res = 0;

	if (DeepDriveSimulation::getInstance())
	{
		PyObject *srcList = 0;
		int32 ok = PyArg_ParseTuple(args, "O!", &PyList_Type, &srcList);
		if (ok)
		{
			std::vector<uint32> agentIds;
			int32 n = PyList_Size(srcList);
			for (int32 i = 0; i < n; ++i)
			{
				PyObject *pItem = PyList_GetItem(srcList, i);
				if (!PyLong_Check(pItem))
				{
					PyErr_SetString(PyExc_TypeError, "List items must be integers");
					return NULL;
				}
				const uint32 id = PyLong_AsLong(pItem);
				agentIds.push_back(id);
			}
			const int32 initRes = DeepDriveSimulation::getInstance()->releaseControl(agentIds);
			if (initRes >= 0)
				res = static_cast<uint32>(initRes);
			else
				return handleError(initRes);
		}
		else
			std::cout << "Wrong arguments\n";
	}
	else
		std::cout << "Not connect to Simulation server\n";

	return Py_BuildValue("i", res);
}

/*	Set control values for a set of specified agent
 *
 *	@param	[{'id', 'steering', 'throttle', 'brake', 'handbrake'}]	List of control values for particular agents
 *
 *	@return	True, if successfully, otherwise false
*/
static PyObject *multi_agent_set_control_values(PyObject *self, PyObject *args)
{
	uint32 res = 0;

	if (DeepDriveSimulation::getInstance())
	{
		PyObject *srcList = 0;
		int32 ok = PyArg_ParseTuple(args, "O!", &PyList_Type, &srcList);
		if (ok)
		{
			std::vector<deepdrive::server::SetControlValuesRequest::ControlValueSet> controlValues;
			int32 n = PyList_Size(srcList);
			std::cout << "set_control_values " << n << "\n";
			for (int32 i = 0; i < n; ++i)
			{
				PyObject *pItem = PyList_GetItem(srcList, i);
				if (pItem && PyDict_Check(pItem))
				{
					std::cout << "got item\n";
					PyObject *pId = PyDict_GetItemString(pItem, "id");
					std::cout << "pId " << pId << "\n";
					if (pId && PyLong_Check(pId))
					{
						std::cout << "got id\n";
						float steering = 0.0f;
						float throttle = 0.0f;
						float brake = 0.0f;
						uint32 handbrake = 0;

						PyObject *pFloat = PyDict_GetItemString(pItem, "steering");
						if (pFloat && PyFloat_Check(pFloat))
							steering = static_cast<float> (PyFloat_AsDouble(pFloat));

						pFloat = PyDict_GetItemString(pItem, "throttle");
						if (pFloat && PyFloat_Check(pFloat))
							throttle = static_cast<float>(PyFloat_AsDouble(pFloat));

						pFloat = PyDict_GetItemString(pItem, "brake");
						if (pFloat && PyFloat_Check(pFloat))
							brake = static_cast<float>(PyFloat_AsDouble(pFloat));

						PyObject *pInt = PyDict_GetItemString(pItem, "handbrake");
						if (pInt && PyLong_Check(pInt))
							handbrake = PyLong_AsLong(pInt) == 0 ? 0 : 1;

						std::cout << PyLong_AsLong(pId) << " s " << steering << " t " << throttle << " b " << brake << " hb " << handbrake << "\n";

						controlValues.push_back(deepdrive::server::SetControlValuesRequest::ControlValueSet(PyLong_AsLong(pId), steering, throttle, brake, handbrake));
					}
					else
					{
						PyErr_SetString(PyExc_TypeError, "Id must be integer");
						return NULL;
					}
				}
				else
				{
					PyErr_SetString(PyExc_TypeError, "List items must be dictionary");
					return NULL;
				}
			}
			const int32 initRes = DeepDriveSimulation::getInstance()->setControlValues(controlValues);
			if (initRes >= 0)
				res = static_cast<uint32>(initRes);
			else
				return handleError(initRes);
		}
		else
			std::cout << "Wrong arguments\n";
	}
	else
		std::cout << "Not connect to Simulation server\n";

	return Py_BuildValue("i", res);
}



/*	Set control values for a set of specified agent
 *
 *	@return	A list of snapshots for all remotely controlled agents.
*/
static PyObject *multi_agent_step(PyObject *self, PyObject *args)
{
	uint32 res = 0;
	if (DeepDriveSimulation::getInstance())
	{
		std::vector<PyMultiAgentSnapshotObject*> snapshots;
		const int32 sentRes = DeepDriveSimulation::getInstance()->step(snapshots);
		if (sentRes >= 0)
			res = static_cast<uint32>(sentRes);
		else
			return handleError(sentRes);
	}
	else
		std::cout << "Not connect to Simulation server\n";

	return Py_BuildValue("i", res);
}

static PyMethodDef MultiAgentMethods[] =	{	{"get_agents_list", (PyCFunction) multi_agent_get_agents_list, METH_VARARGS, "Retrieve a list of available agents"}
											,	{"request_control", (PyCFunction) multi_agent_request_control, METH_VARARGS, "Request remote control for a list of agents"}
											,	{"release_control", (PyCFunction) multi_agent_release_control, METH_VARARGS, "Release remote control for a list of agents"}
											,	{"set_control_values", (PyCFunction) multi_agent_set_control_values, METH_VARARGS, "Set control values for a list of agents"}
											,	{"step", (PyCFunction) multi_agent_step, METH_VARARGS, "Set control values for a set of specified agent"}
											// ,	{"reset_multi_agent", (PyCFunction) reset_multi_agent, METH_VARARGS | METH_KEYWORDS, "Reset multi_agent"}
											// ,	{"set_graphics_settings", set_graphics_settings, METH_VARARGS, "Set graphics settings"}
											// ,	{"set_date_and_time", (PyCFunction) set_date_and_time, METH_VARARGS | METH_KEYWORDS, "Set date and time for multi_agent"}
											// ,	{"set_sun_multi_agent_speed", (PyCFunction) set_sun_multi_agent_speed, METH_VARARGS | METH_KEYWORDS, "Set speed for sun multi_agent"}
											,	{NULL,     NULL,             0,            NULL}        /* Sentinel */
											};

static struct PyModuleDef multi_agent_module = {
		PyModuleDef_HEAD_INIT,
		"multi_agent",   /* name of module */
		NULL,          /* module documentation, may be NULL */
		-1,            /* size of per-interpreter state of the module,
				          or -1 if the module keeps state in global variables. */
		MultiAgentMethods
};

PyMODINIT_FUNC PyInit_multi_agent(void)
{
	if (PyType_Ready(&PyMultiAgentSnapshotType) < 0)
		return 0;

	PyObject *m = PyModule_Create(&multi_agent_module);
	if (m)
	{
	}

	std::cout << "## ><>|><> PyInit_multi_agent <><|<>< ##\n";

	return m;
}

PyObject* create_multi_agent_module()
{
	return PyInit_multi_agent();
}
