#include "Python.h"

#include "Engine.h"

#include "deepdrive_simulation/DeepDriveSimulation.hpp"
#include "common/ClientErrorCode.hpp"

#include "deepdrive_simulation/PySimulationGraphicsSettingsObject.h"

#include <iostream>

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
	PyObject *ret = PyDict_New();

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
			std::cout << "Address successfully parsed " << ip4Address.toStr(true) << "\n";
			std::cout << "Seed " <<  seed << " time dilation " << timeDilation << "\n";
			std::cout << "gfx settings " <<  graphicsSettings << "\n";

#if 0
			DeepDriveClient *client = new DeepDriveClient(ip4Address);
			if	(	client
				&& 	client->isConnected()
				)
			{
				std::cout << "Successfully connected to " << ip4Address.toStr(true) << "\n";
				deepdrive::server::RegisterClientResponse registerClientResponse;
				const int32 res = client->registerClient(registerClientResponse, request_master_role, seed, timeDilation, startLocation, reinterpret_cast<PySimulationGraphicsSettingsObject*> (graphicsSettings));
				if(res >= 0)
				{
					uint32 clientId = registerClientResponse.client_id;
					std::cout << "Client id is " << std::to_string(clientId) << "\n";
					PyDict_SetItem(ret, PyUnicode_FromString("client_id"), PyLong_FromUnsignedLong(clientId));
					if(clientId)
					{
						addClient(clientId, client);
						PyDict_SetItem(ret, PyUnicode_FromString("granted_master_role"),
							PyLong_FromUnsignedLong(client->m_isMaster));
						PyDict_SetItem(ret, PyUnicode_FromString("shared_memory_size"),
							PyLong_FromUnsignedLong(client->m_SharedMemorySize));
						PyDict_SetItem(ret, PyUnicode_FromString("max_supported_cameras"),
							PyLong_FromUnsignedLong(client->m_MaxSupportedCameras));
						PyDict_SetItem(ret, PyUnicode_FromString("max_capture_resolution"),
							PyLong_FromUnsignedLong(client->m_MaxCaptureResolution));
						PyDict_SetItem(ret, PyUnicode_FromString("inactivity_timeout_ms"),
						 	PyLong_FromUnsignedLong(client->m_InactivityTimeout));
						PyDict_SetItem(ret, PyUnicode_FromString("shared_memory_name"),
							PyUnicode_FromString(client->m_SharedMemoryName.c_str()));
						PyDict_SetItem(ret, PyUnicode_FromString("server_protocol_version"),
							PyUnicode_FromString(client->m_ServerProtocolVersion.c_str()));
					}
				}
				else
					return handleError(res);
			}
			else
				std::cout << "Couldn't connect to " << ip4Address.toStr(true) << "\n";
#endif
		}
		else
			std::cout << ipStr << " doesnt appear to be a valid IP4 address\n";
	}
	else
		std::cout << "Wrong arguments\n";
	return ret;
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
 *	@param	uint32		Client id
 *	@param	uint32		Year
 *	@param	uint32		Month
 *	@param	uint32		Day
 *	@param	uint32		Hour
 *	@param	uint32		Minute
 *	@return	True, if successfully, otherwise false
*/
static PyObject* set_simulation_date_and_time(PyObject *self, PyObject *args, PyObject *keyWords)
{
	uint32 res = 0;
#if 0

	uint32 clientId = 0;
	uint32 year = 2011;
	uint32 month = 8;
	uint32 day = 1;
	uint32 hour = 11;
	uint32 minute = 30;

	char *keyWordList[] = {"year", "month", "day", "hour", "minute", NULL};
	int32 ok = PyArg_ParseTupleAndKeywords(args, keyWords, "I|IIIII", keyWordList, &clientId, &year, &month, &day, &hour, &minute);
	if(ok)
	{
		DeepDriveClient *client = getClient(clientId);
		if	(	client
			&&	client->isConnected()
			)
		{
			const int32 initRes = 1;// DeepDriveSimulation::setSunSimulation(*client, month, day, minute, hour, speed);
			if(initRes >= 0)
				res = static_cast<uint32> (initRes);
			else
				return handleError(initRes);
		}
	}
	else
		std::cout << "Wrong arguments\n";
#endif
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
												,	{"set_simulation_date_and_time", (PyCFunction) set_simulation_date_and_time, METH_VARARGS | METH_KEYWORDS, "Set date and time for simulation"}
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
