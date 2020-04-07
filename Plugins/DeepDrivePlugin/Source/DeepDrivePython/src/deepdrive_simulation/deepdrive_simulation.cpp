#include "Python.h"

#include "Engine.h"

#include "deepdrive_simulation/DeepDriveSimulation.hpp"
#include "deepdrive_simulation/deepdrive_simulation_error.h"

#include "deepdrive_simulation/PySimulationGraphicsSettingsObject.h"

#include <iostream>


/*	Create a new client, tries to connect to specified DeepDriveServer
 *
 *	@param	address		IP4 address of server
 *	@param	uint32		Port
 *	@param	uint32		Seed
 *	@param	number		Global time dilation
 *	@param	number		Agent starting location
 *	@param	object		Graphics settings
 *
 *	@return	Client id, 0 in case of error
*/
static PyObject* deepdrive_simulation_connect(PyObject *self, PyObject *args, PyObject *keyWords)
{
	uint32 res = 0;
	const char *ipStr;
	uint32 port = 0;
	uint32 seed = 0;
	float timeDilation = 1.0f;
	float startLocation = -1.0f;
	PyObject *graphicsSettings = 0;

	char *keyWordList[] = {"ip_address", "port", "seed", "time_dilation", "agent_start_location", "graphics_settings", NULL};
	int32 ok = PyArg_ParseTupleAndKeywords(args, keyWords, "sI|IffO!", keyWordList, &ipStr, &port, &seed, &timeDilation, &startLocation, &PySimulationGraphicsSettingsType, &graphicsSettings);
	if(ok && port > 0 && port < 65536)
	{
		IP4Address ip4Address;

		if(ip4Address.set(ipStr, port))
		{
			if(DeepDriveSimulation::getInstance() == 0)
			{
				DeepDriveSimulation::create(ip4Address);
				std::cout << "Created DeepDriveSimulation\n";

				if(DeepDriveSimulation::getInstance() && DeepDriveSimulation::getInstance()->isConnected())
				{
					const int32 configureRes = DeepDriveSimulation::getInstance()->configureSimulation(seed, timeDilation, startLocation, reinterpret_cast<PySimulationGraphicsSettingsObject*> (graphicsSettings));
					if(configureRes < 0)
						handleError(configureRes);
					else
						res = 1;
				}

			}
			else
				std::cout << "Already connected\n";
		}
		else
			std::cout << ipStr << " doesn't appear to be a valid IP4 address\n";
	}
	else
		std::cout << "Wrong arguments\n";

	return Py_BuildValue("i", res);
}

/*	Disconnect to simulation server
 *
*/
static PyObject* deepdriuve_simulation_disconnect(PyObject *self, PyObject *args)
{
	uint32 res = 0;
	DeepDriveSimulation::destroy();

	return Py_BuildValue("i", res);
}

/*	Reset simulation
 *
 *	@param	number		Global time dilation
 *	@param	number		Agent starting location
 *
*/
static PyObject* reset_simulation(PyObject *self, PyObject *args, PyObject *keyWords)
{
	uint32 res = 0;

	if(DeepDriveSimulation::getInstance())
	{
		float timeDilation = 1.0f;
		float startLocation = -1.0f; 

		char *keyWordList[] = {"time_dilation", "agent_start_location", NULL};
		int32 ok = PyArg_ParseTupleAndKeywords(args, keyWords, "|ff", keyWordList, &timeDilation, &startLocation);
		if(ok)
		{
			const int32 requestRes = DeepDriveSimulation::getInstance()->resetSimulation(timeDilation, startLocation);
			if(requestRes >= 0)
				res = static_cast<uint32> (requestRes);
			else
				return handleError(requestRes);
		}
		else
		{
			std::cout << "Wrong arguments\n";
		}
	}
	else
		std::cout << "Not connect to Simulation server\n";

	return Py_BuildValue("i", res);
}

/*	Set graphics settings
 *
 *	@param	object		Graphics settings
 *
*/
static PyObject* set_graphics_settings(PyObject *self, PyObject *args)
{
	uint32 res = 0;

	if(DeepDriveSimulation::getInstance())
	{
		PyObject *graphicsSettings = 0;

		int32 ok = PyArg_ParseTuple(args, "O!", &PySimulationGraphicsSettingsType, &graphicsSettings);
		if(ok)
		{
			const int32 requestRes = DeepDriveSimulation::getInstance()->setGraphicsSettings(reinterpret_cast<PySimulationGraphicsSettingsObject*> (graphicsSettings));
			if(requestRes >= 0)
				res = static_cast<uint32> (requestRes);
			else
				return handleError(requestRes);
		}
		else
		{
			std::cout << "Wrong arguments\n";
		}
	}
	else
		std::cout << "Not connect to Simulation server\n";

	return Py_BuildValue("i", res);
}


/*	Set date and time for simulation
 *
 *	@param	uint32		Year
 *	@param	uint32		Month
 *	@param	uint32		Day
 *	@param	uint32		Hour
 *	@param	uint32		Minute
 *	@return	True, if successfully, otherwise false
*/
static PyObject* set_date_and_time(PyObject *self, PyObject *args, PyObject *keyWords)
{
	uint32 res = 0;

	if(DeepDriveSimulation::getInstance())
	{
		uint32 year = 2011;
		uint32 month = 8;
		uint32 day = 1;
		uint32 hour = 11;
		uint32 minute = 30;

		char *keyWordList[] = {"year", "month", "day", "hour", "minute", NULL};
		int32 ok = PyArg_ParseTupleAndKeywords(args, keyWords, "|IIIII", keyWordList, &year, &month, &day, &hour, &minute);
		if(ok)
		{
			const int32 requestRes = DeepDriveSimulation::getInstance()->setDateAndTime(year, month, day, minute, hour);
			if(requestRes >= 0)
				res = static_cast<uint32> (requestRes);
			else
				return handleError(requestRes);
		}
		else
			std::cout << "Wrong arguments\n";
	}
	else
		std::cout << "Not connect to Simulation server\n";

	return Py_BuildValue("i", res);
}

/*	Set date and time for simulation
 *
 *	@param	uint32		Speed in simulation seconds per real-time seconds
 *	@return	True, if successfully, otherwise false
*/
static PyObject* set_sun_simulation_speed(PyObject *self, PyObject *args, PyObject *keyWords)
{
	uint32 res = 0;

	if(DeepDriveSimulation::getInstance())
	{
		uint32 speed = 0;

		char *keyWordList[] = {"speed", NULL};
		int32 ok = PyArg_ParseTupleAndKeywords(args, keyWords, "|I", keyWordList, &speed);
		if(ok)
		{
			const int32 initRes = DeepDriveSimulation::getInstance()->setSpeed(speed);
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

static PyMethodDef DeepDriveClientMethods[] =	{	{"connect", (PyCFunction) deepdrive_simulation_connect, METH_VARARGS | METH_KEYWORDS, "Connect to simulation server"}
												,	{"disconnect", (PyCFunction) deepdriuve_simulation_disconnect, METH_VARARGS | METH_KEYWORDS, "Disconnects from simulation server"}
												,	{"reset_simulation", (PyCFunction) reset_simulation, METH_VARARGS | METH_KEYWORDS, "Reset simulation"}
												,	{"set_graphics_settings", set_graphics_settings, METH_VARARGS, "Set graphics settings"}
												,	{"set_date_and_time", (PyCFunction) set_date_and_time, METH_VARARGS | METH_KEYWORDS, "Set date and time for simulation"}
												,	{"set_sun_simulation_speed", (PyCFunction) set_sun_simulation_speed, METH_VARARGS | METH_KEYWORDS, "Set speed for sun simulation"}
												,	{NULL,     NULL,             0,            NULL}        /* Sentinel */
												};

static struct PyModuleDef deepdrive_simulation_module = {
		PyModuleDef_HEAD_INIT,
		"deepdrive_simulation",   /* name of module */
		NULL,          /* module documentation, may be NULL */
		-1,            /* size of per-interpreter state of the module,
				          or -1 if the module keeps state in global variables. */
		DeepDriveClientMethods
};

extern PyObject* create_multi_agent_module();

PyMODINIT_FUNC PyInit_deepdrive_simulation(void)
{
	// if (PyType_Ready(&PyDeepDriveClientRegisterClientRequestType) < 0)
	//	return 0;

	std::cout << "#### ><> ><> PyInit_deepdrive_simulation <>< <>< ####\n";

	if (PyType_Ready(&PySimulationGraphicsSettingsType) < 0)
		return 0;

	PyObject *m  = PyModule_Create(&deepdrive_simulation_module);
	if (m)
	{
		PyObject *mam = create_multi_agent_module();
		Py_INCREF(mam);
		PyModule_AddObject(m, "multi_agent", mam);

		setupErrorTypes(m);

		Py_INCREF(&PySimulationGraphicsSettingsType);
		PyModule_AddObject(m, "SimulationGraphicsSettings", (PyObject *)&PySimulationGraphicsSettingsType);

	}

	return m;
}



