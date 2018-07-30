#include "Python.h"

#include "Engine.h"

#include "deepdrive_simulation/DeepDriveSimulation.hpp"
#include "common/ClientErrorCode.hpp"

#include "deepdrive_simulation/PySimulationGraphicsSettingsObject.h"

#include <iostream>

static DeepDriveSimulation *g_DeepDriveSimulation = 0;

static PyObject *DeepDriveClientError;
static PyObject *ConnectionLostError;
static PyObject *NotConnectedError;
static PyObject *TimeOutError;
static PyObject *ClientDoesntExistError;
static PyObject *UnknownError;

static PyObject* handleError(int32 errorCode)
{
	if(errorCode == ClientErrorCode::NOT_CONNECTED)
	{
		PyErr_SetString(ClientDoesntExistError, "Client doesn't exist");
	}
	else if(errorCode == ClientErrorCode::CONNECTION_LOST)
	{
		PyErr_SetString(ConnectionLostError, "Connection to server lost");
	}
	else if(errorCode == ClientErrorCode::TIME_OUT)
	{
		PyErr_SetString(TimeOutError, "Network time out");
	}
	else if(errorCode == ClientErrorCode::UNKNOWN_ERROR)
	{
		PyErr_SetString(UnknownError, "Unknown network error");
	}
	return 0;
}

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
			if(g_DeepDriveSimulation == 0)
			{
				g_DeepDriveSimulation = new DeepDriveSimulation(ip4Address);
				std::cout << "Created DeepDriveSimulation\n";

				if(g_DeepDriveSimulation && g_DeepDriveSimulation->isConnected())
				{
					const int32 configureRes = g_DeepDriveSimulation->configureSimulation(seed, timeDilation, startLocation, reinterpret_cast<PySimulationGraphicsSettingsObject*> (graphicsSettings));
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
	if(g_DeepDriveSimulation)
	{
		delete g_DeepDriveSimulation;
		g_DeepDriveSimulation = 0;
	}


	return 0;
}

/*	Reset simulation
 *
 *	@param	uint32		Client Id
 *	@param	number		Global time dilation
 *	@param	number		Agent starting location
 *	@param	object		Graphics settings
 *
*/
static PyObject* reset_simulation(PyObject *self, PyObject *args, PyObject *keyWords)
{
#if 0
	uint32 clientId = 0;

	float timeDilation = 1.0f;
	float startLocation = -1.0f; 
	PyObject *graphicsSettings = 0;

	char *keyWordList[] = {"client_id", "time_dilation", "agent_start_location", "graphics_settings", NULL};
	int32 ok = PyArg_ParseTupleAndKeywords(args, keyWords, "I|ffO!", keyWordList, &clientId, &timeDilation, &startLocation, &PySimulationGraphicsSettingsType, &graphicsSettings);
	if(ok)
	{
		std::cout << "Reset simulation clientId " << clientId << " \n";
		DeepDriveClient *client = getClient(clientId);
		if(client)
		{
			DeepDriveSimulation::resetSimulation(*client, timeDilation, startLocation, reinterpret_cast<PySimulationGraphicsSettingsObject*> (graphicsSettings));
		}
		else
		{
			PyErr_SetString(ClientDoesntExistError, "Client doesn't exist");
			return 0;
		}
	}
	else
	{
		std::cout << "Wrong arguments\n";
	}
#endif
	return Py_BuildValue("");
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

	if(g_DeepDriveSimulation)
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
			const int32 initRes = g_DeepDriveSimulation->setDateAndTime(year, month, day, minute, hour);
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

/*	Set date and time for simulation
 *
 *	@param	uint32		Client id
 *	@param	uint32		Year
 *	@param	uint32		Month
 *	@param	uint32		Day
 *	@param	uint32		Hour
 *	@param	uint32		Minute
 *	@return	True, if successfully, otherwise false
*/
static PyObject* set_sun_simulation_speed(PyObject *self, PyObject *args)
{
	uint32 res = 0;

	return Py_BuildValue("i", res);
}

static PyMethodDef DeepDriveClientMethods[] =	{	{"connect", (PyCFunction) deepdrive_simulation_connect, METH_VARARGS | METH_KEYWORDS, "Creates a new client which tries to connect to DeepDriveServer"}
												,	{"reset_simulation", (PyCFunction) reset_simulation, METH_VARARGS | METH_KEYWORDS, "Reset simulation"}
												,	{"set_date_and_time", (PyCFunction) set_date_and_time, METH_VARARGS | METH_KEYWORDS, "Set date and time for simulation"}
												,	{"set_sun_simulation_speed", (PyCFunction) set_sun_simulation_speed, METH_VARARGS, "Set speed for sun simulation"}
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

PyMODINIT_FUNC PyInit_deepdrive_simulation(void)
{
	// if (PyType_Ready(&PyDeepDriveClientRegisterClientRequestType) < 0)
	//	return 0;

	std::cout << "###### ><> ><> PyInit_deepdrive_simulation <>< <>< ######\n";

	if (PyType_Ready(&PySimulationGraphicsSettingsType) < 0)
		return 0;


	PyObject *m  = PyModule_Create(&deepdrive_simulation_module);
	if (m)
	{
		DeepDriveClientError = PyErr_NewException("deepdrive_simulation.error", NULL, NULL);
		Py_INCREF(DeepDriveClientError);
		PyModule_AddObject(m, "error", DeepDriveClientError);

		ConnectionLostError = PyErr_NewException("deepdrive_simulation.connection_lost", NULL, NULL);
		Py_INCREF(ConnectionLostError);
		PyModule_AddObject(m, "connection_lost", ConnectionLostError);

		NotConnectedError = PyErr_NewException("deepdrive_simulation.not_connected", NULL, NULL);
		Py_INCREF(NotConnectedError);
		PyModule_AddObject(m, "not_connected", NotConnectedError);

		TimeOutError = PyErr_NewException("deepdrive_simulation.time_out", NULL, NULL);
		Py_INCREF(TimeOutError);
		PyModule_AddObject(m, "time_out", TimeOutError);

		ClientDoesntExistError = PyErr_NewException("deepdrive_simulation.client_doesnt_exist", NULL, NULL);
		Py_INCREF(ClientDoesntExistError);
		PyModule_AddObject(m, "client_doesnt_exist", ClientDoesntExistError);

		UnknownError = PyErr_NewException("deepdrive_simulation.unknown_error", NULL, NULL);
		Py_INCREF(UnknownError);
		PyModule_AddObject(m, "unknown_error", UnknownError);

		Py_INCREF(&PySimulationGraphicsSettingsType);
		PyModule_AddObject(m, "SimulationGraphicsSettings", (PyObject *)&PySimulationGraphicsSettingsType);

	}

	return m;
}
