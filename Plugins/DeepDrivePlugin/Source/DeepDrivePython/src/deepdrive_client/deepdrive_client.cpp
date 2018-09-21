#include "Python.h"

#include "Engine.h"

#include "deepdrive_client/DeepDriveClient.hpp"
#include "deepdrive_client/DeepDriveClientMap.hpp"
#include "common/ClientErrorCode.hpp"

#include "deepdrive_simulation/DeepDriveSimulation.hpp"
#include "deepdrive_simulation/PySimulationGraphicsSettingsObject.h"

#include "common/NumPyUtils.h"

#include "numpy/arrayobject.h"

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
 *	@param	uint32		Request master role
 *
 *	@return	Client id, 0 in case of error
*/
static PyObject* deepdrive_client_create(PyObject *self, PyObject *args, PyObject *keyWords)
{
	deepdrive::server::RegisterClientResponse res;

	PyObject *ret = PyDict_New();

	const char *ipStr;
	uint32 port = 0;
	bool request_master_role = true;

	char *keyWordList[] = {"ip_address", "port", "request_master_role", NULL};
	int32 ok = PyArg_ParseTupleAndKeywords(args, keyWords, "sI|p", keyWordList, &ipStr, &port, &request_master_role);
	if(ok && port > 0 && port < 65536)
	{
		IP4Address ip4Address;

		if(ip4Address.set(ipStr, port))
		{
			std::cout << "Address successfully parsed " << ip4Address.toStr(true) << "\n";

			DeepDriveClient *client = new DeepDriveClient(ip4Address);
			if	(	client
				&& 	client->isConnected()
				)
			{
				std::cout << "Successfully connected to " << ip4Address.toStr(true) << "\n";
				deepdrive::server::RegisterClientResponse registerClientResponse;
				const int32 res = client->registerClient(registerClientResponse, request_master_role);
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
		}
		else
			std::cout << ipStr << " doesnt appear to be a valid IP4 address\n";
	}
	else
		std::cout << "Wrong arguments\n";

	return ret;
}

/*	Close an existing client
 *
 *	@param	uint32		Client Id
*/
static PyObject* deepdrive_client_close(PyObject *self, PyObject *args)
{
	uint32 res = 0;

	uint32 clientId = 0;
	int32 ok = PyArg_ParseTuple(args, "i", &clientId);

	if(ok && clientId > 0)
	{
		if(!removeClient(clientId))
		{
			PyErr_SetString(ClientDoesntExistError, "Client doesn't exist");
			return 0;
		}
	}


	return Py_BuildValue("i", res);
}

/*	Send Keep-alive message to server
 *
 *	@param	uint32		Client Id
*/
static PyObject* deepdrive_client_keep_alive(PyObject *self, PyObject *args)
{
	uint32 res = 0;
	// client.keepAlive()
	return Py_BuildValue("i", res);
}


/*	Register a new capture camera
 *
 *	@param	uint32		Client Id
 *	@param	number		Horizontal field of view
 *	@param	number		Width
 *	@param	number		Height
 *	@param	vector		Relative position
 *	@param	vector		Relative rotation
 *	@param	string		Label of camera
 *
 *	@return	uint32		Camera Id or 0 when not successful
*/
static PyObject* deepdrive_client_register_camera(PyObject *self, PyObject *args, PyObject *keyWords)
{
	int32 res = 0;

	uint32 clientId = 0;
	float hFoV = 1.0f;
	int32 captureWidth = 0;
	int32 captureHeight = 0;

	PyObject *relPosPtr = 0;
	PyObject *relRotPtr = 0;

	const char *label = 0;

	char *keyWordList[] = {"client_id", "field_of_view", "capture_width", "capture_height", "relative_position", "relative_rotation", "label", NULL};
	int32 ok = PyArg_ParseTupleAndKeywords(args, keyWords, "I|fIIOOs", keyWordList, &clientId, &hFoV, &captureWidth, &captureHeight, &relPosPtr, &relRotPtr, &label);

	if(ok)
	{
		std::cout << "Register camera for client " << clientId << " " << captureWidth << "x" << captureHeight << " FoV " << hFoV << " label " << (label ? label : "UNKNOWN") << "\n";

		float relPos[3] = {0.0f, 0.0f, 0.0f};
		float relRot[3] = {0.0f, 0.0f, 0.0f};
		if	(	(relPosPtr == 0 || NumPyUtils::getVector3(relPosPtr, relPos, PyObject_TypeCheck(relPosPtr, &PyArray_Type) != 0))
			&&	(relRotPtr == 0 || NumPyUtils::getVector3(relRotPtr, relRot, PyObject_TypeCheck(relRotPtr, &PyArray_Type) != 0))
			)
		{
			DeepDriveClient *client = getClient(clientId);
			if(client)
			{
				res = client->registerCamera(hFoV, captureWidth, captureHeight, relPos, relRot, label);
				if(res < 0)
					return handleError(res);
			}
			else
			{
				PyErr_SetString(ClientDoesntExistError, "Client doesn't exist");
				return 0;
			}
		}
	}
	else
		std::cout << "Wrong arguments\n";

	return Py_BuildValue("i", res);
}

/*	Register a new capture camera
 *
 *	@param	uint32		Client Id
 *	@param	uint32		Camera Id
*/
static PyObject* deepdrive_client_unregister_camera(PyObject *self, PyObject *args, PyObject *keyWords)
{
	int32 res = 0;

	uint32 clientId = 0;
	uint32 cameraId = 0;

	char *keyWordList[] = {"client_id", "camera_id", NULL};
	int32 ok = PyArg_ParseTupleAndKeywords(args, keyWords, "I|I", keyWordList, &clientId, &cameraId);

	if(ok)
	{
		std::cout << "Unregister camera " << cameraId << " for client " << clientId << "\n";
		DeepDriveClient *client = getClient(clientId);
		if(client)
		{
			res = client->unregisterCamera(cameraId);
			if(res < 0)
				return handleError(res);
		}
		else
		{
			PyErr_SetString(ClientDoesntExistError, "Client doesn't exist");
			return 0;
		}
	}
	else
		std::cout << "Wrong arguments\n";

	return Py_BuildValue("i", 1);
}


/*	Request agent control
 *
 *	@param	uint32		Client Id
 *
 *	@return	True, if conmtrol was granted, otherwise false
*/
static PyObject* deepdrive_client_request_agent_control(PyObject *self, PyObject *args)
{
	uint32 clientId = 0;
	int32 ok = PyArg_ParseTuple(args, "i", &clientId);

	int32 res = 0;

	if(ok && clientId > 0)
	{
		DeepDriveClient *client = getClient(clientId);
		if(client)
		{
			res = client->requestAgentControl();
			std::cout << "requestAgentControl res " << res << "\n";
			if(res < 0)
			{
			    std::cout << "handle error res " << res << "\n";
				return handleError(res);
			}
		}
		else
		{
			PyErr_SetString(ClientDoesntExistError, "Client doesn't exist");
			return 0;
		}
	}

    std::cout << "res in deepdrive_client_request_agent_control" << res << "\n";

	return Py_BuildValue("i", res);
}


/*	Release agent control
 *
 *	@param	uint32		Client Id
 *
*/
static PyObject* deepdrive_client_release_agent_control(PyObject *self, PyObject *args)
{
	uint32 clientId = 0;
	int32 ok = PyArg_ParseTuple(args, "i", &clientId);

	if(ok && clientId > 0)
	{
		DeepDriveClient *client = getClient(clientId);
		if(client)
		{
			const int32 res = client->releaseAgentControl();
			if(res < 0)
				return handleError(res);
		}
		else
		{
			PyErr_SetString(ClientDoesntExistError, "Client doesn't exist");
			return 0;
		}
	}

	return Py_BuildValue("");
}

/*	Reset agent
 *
 *	@param	uint32		Client Id
 *
*/
static PyObject* deepdrive_client_reset_agent(PyObject *self, PyObject *args)
{
	uint32 clientId = 0;
	int32 ok = PyArg_ParseTuple(args, "i", &clientId);

	if(ok && clientId > 0)
	{
		DeepDriveClient *client = getClient(clientId);
		if(client)
		{
			const int32 res = client->resetAgent();
			if(res < 0)
				return handleError(res);
		}
		else
		{
			PyErr_SetString(ClientDoesntExistError, "Client doesn't exist");
			return 0;
		}
	}

	return Py_BuildValue("");
}


/*	Send control values to server
 *
 *	@param	uint32		Client Id
 *	@param	number		Steering
 *	@param	number		Throttle
 *	@param	number		Brake
 *	@param	uint32		Handbrake
*/
static PyObject* deepdrive_client_set_control_values(PyObject *self, PyObject *args, PyObject *keyWords)
{
	uint32 clientId = 0;
	float steering = 0.0f;
	float throttle = 0.0f;
	float brake = 0.0f;
	uint32 handbrake = 0;

	char *keyWordList[] = {"client_id", "steering", "throttle", "brake", "handbrake", NULL};
	int32 ok = PyArg_ParseTupleAndKeywords(args, keyWords, "I|fffI", keyWordList, &clientId, &steering, &throttle, &brake, &handbrake);

	if(ok)
	{
		DeepDriveClient *client = getClient(clientId);
		if(client)
		{
			const int32 res = client->setControlValues(steering, throttle, brake, handbrake);
			if(res < 0)
				return handleError(res);
		}
		else
		{
			PyErr_SetString(ClientDoesntExistError, "Client doesn't exist");
			return 0;
		}
	}
	else
		std::cout << "Wrong arguments\n";

	return Py_BuildValue("");
}

/*	Activate synchronous stepping mode on server
 *
 *	@param	uint32		Client Id
*/
static PyObject* deepdrive_client_activate_synchronous_stepping(PyObject *self, PyObject *args, PyObject *keyWords)
{
	uint32 clientId = 0;
	int32 ok = PyArg_ParseTuple(args, "i", &clientId);

	int32 res = 0;

	if(ok && clientId > 0)
	{
		DeepDriveClient *client = getClient(clientId);
		if(client)
		{
			res = client->activateSynchronousStepping();
			if(res < 0)
				return handleError(res);
		}
		else
		{
			PyErr_SetString(ClientDoesntExistError, "Client doesn't exist");
			return 0;
		}
	}

	return Py_BuildValue("i", res);
}

/*	Deactivate synchronous stepping mode on server
 *
 *	@param	uint32		Client Id
*/
static PyObject* deepdrive_client_deactivate_synchronous_stepping(PyObject *self, PyObject *args, PyObject *keyWords)
{
	uint32 clientId = 0;
	int32 ok = PyArg_ParseTuple(args, "i", &clientId);

	if(ok && clientId > 0)
	{
		DeepDriveClient *client = getClient(clientId);
		if(client)
		{
			const int32 res = client->deactivateSynchronousStepping();
			if(res < 0)
				return handleError(res);
		}
		else
		{
			PyErr_SetString(ClientDoesntExistError, "Client doesn't exist");
			return 0;
		}
	}

	return Py_BuildValue("");
}

/*	Advance server by specified time step.
 *  This is a blocking call waiting until server has advanced so be carefully with size of time step
 *
 *	@param	uint32		Client Id
 *	@param	number		Time Delta
 *	@param	number		Steering
 *	@param	number		Throttle
 *	@param	number		Brake
 *	@param	uint32		Handbrake
*/
static PyObject* deepdrive_client_advance_synchronous_stepping(PyObject *self, PyObject *args, PyObject *keyWords)
{
	uint32 clientId = 0;
	float dT = 0.0f;
	float steering = 0.0f;
	float throttle = 0.0f;
	float brake = 0.0f;
	uint32 handbrake = 0;

	char *keyWordList[] = {"client_id", "time_step", "steering", "throttle", "brake", "handbrake", NULL};
	int32 ok = PyArg_ParseTupleAndKeywords(args, keyWords, "I|ffffI", keyWordList, &clientId, &dT, &steering, &throttle, &brake, &handbrake);

	int32 seqNr = -1;

	if(ok)
	{
		DeepDriveClient *client = getClient(clientId);
		if(client)
		{
			const int32 res = client->advanceSynchronousStepping(dT, steering, throttle, brake, handbrake);
			if(res < 0)
				return handleError(res);
			else
				seqNr = res;
		}
		else
		{
			PyErr_SetString(ClientDoesntExistError, "Client doesn't exist");
			return 0;
		}
	}
	else
		std::cout << "Wrong arguments\n";

	return Py_BuildValue("i", seqNr);
}

/*	Send control values to server
 *
 *	@param	uint32		Client Id
 *
 *	@return [shared memory name, shared memory size] or None in case of an error
*/
static PyObject* deepdrive_client_get_shared_memory(PyObject *self, PyObject *args)
{
	uint32 clientId = 0;
	int32 ok = PyArg_ParseTuple(args, "i", &clientId);

	if(ok && clientId > 0)
	{
		DeepDriveClient *client = getClient(clientId);
		if(client)
		{
			return Py_BuildValue("(si)", client->getSharedMemoryName(), client->getSharedMemorySize());
		}
	}

	return Py_BuildValue("");
}


/*	Advance server by specified time step.
 *  This is a blocking call waiting until server has advanced so be carefully with size of time step
 *
 *	@param	uint32		Client Id
 *	@param	int32		Camera Id
 *	@param	string		View mode
*/
static PyObject* deepdrive_client_set_view_mode(PyObject *self, PyObject *args, PyObject *keyWords)
{
	uint32 clientId = 0;
	int32 cameraId = -1;
	const char *viewMode = 0;

	char *keyWordList[] = {"client_id", "camera_id", "view_mode", NULL};
	int32 ok = PyArg_ParseTupleAndKeywords(args, keyWords, "I|is", keyWordList, &clientId, &cameraId, &viewMode);

	std::cout << "Set view mode " << clientId << " \n";

	int32 res = 0;
	if(ok)
	{
		DeepDriveClient *client = getClient(clientId);
		if(client)
		{
			res = client->setViewMode(cameraId, viewMode);
			if(res < 0)
				return handleError(res);
		}
		else
		{
			PyErr_SetString(ClientDoesntExistError, "Client doesn't exist");
			return 0;
		}
	}
	else
		std::cout << "Wrong arguments\n";

	return Py_BuildValue("i", res);
}


static PyMethodDef DeepDriveClientMethods[] =	{	{"create", (PyCFunction) deepdrive_client_create, METH_VARARGS | METH_KEYWORDS, "Creates a new client which tries to connect to DeepDriveServer"}
												,	{"close", deepdrive_client_close, METH_VARARGS, "Closes an existing client connection and frees all depending resources"}
												,	{"register_camera", (PyCFunction) deepdrive_client_register_camera, METH_VARARGS | METH_KEYWORDS, "Register a capture camera"}
												,	{"unregister_camera", (PyCFunction) deepdrive_client_unregister_camera, METH_VARARGS | METH_KEYWORDS, "Unregister a capture camera"}
												,	{"get_shared_memory", deepdrive_client_get_shared_memory, METH_VARARGS, "Get shared memory name and size for client"}
												,	{"request_agent_control", deepdrive_client_request_agent_control, METH_VARARGS, "Request control over agent"}
												,	{"release_agent_control", deepdrive_client_release_agent_control, METH_VARARGS, "Release control over agent"}
												,	{"reset_agent", deepdrive_client_reset_agent, METH_VARARGS, "Reset the agent"}
												,	{"set_control_values", (PyCFunction) deepdrive_client_set_control_values, METH_VARARGS | METH_KEYWORDS, "Send control value set to server"}
												,	{"activate_synchronous_stepping", (PyCFunction) deepdrive_client_activate_synchronous_stepping, METH_VARARGS, "Send control value set to server"}
												,	{"deactivate_synchronous_stepping", (PyCFunction) deepdrive_client_deactivate_synchronous_stepping, METH_VARARGS, "Send control value set to server"}
												,	{"advance_synchronous_stepping", (PyCFunction) deepdrive_client_advance_synchronous_stepping, METH_VARARGS | METH_KEYWORDS, "Send control value set to server"}
												,	{"set_view_mode", (PyCFunction) deepdrive_client_set_view_mode, METH_VARARGS | METH_KEYWORDS, "Set specific view mode at a client"}
												,	{NULL,     NULL,             0,            NULL}        /* Sentinel */
												};

static struct PyModuleDef deepdrive_client_module = {
		PyModuleDef_HEAD_INIT,
		"deepdrive_client",   /* name of module */
		NULL,          /* module documentation, may be NULL */
		-1,            /* size of per-interpreter state of the module,
				          or -1 if the module keeps state in global variables. */
		DeepDriveClientMethods
};

PyMODINIT_FUNC PyInit_deepdrive_client(void)
{
	if (PyType_Ready(&PySimulationGraphicsSettingsType) < 0)
		return 0;

	import_array();

	PyObject *m  = PyModule_Create(&deepdrive_client_module);
	if (m)
	{
		DeepDriveClientError = PyErr_NewException("deepdrive_client.error", NULL, NULL);
		Py_INCREF(DeepDriveClientError);
		PyModule_AddObject(m, "error", DeepDriveClientError);

		ConnectionLostError = PyErr_NewException("deepdrive_client.connection_lost", NULL, NULL);
		Py_INCREF(ConnectionLostError);
		PyModule_AddObject(m, "connection_lost", ConnectionLostError);

		NotConnectedError = PyErr_NewException("deepdrive_client.not_connected", NULL, NULL);
		Py_INCREF(NotConnectedError);
		PyModule_AddObject(m, "not_connected", NotConnectedError);

		TimeOutError = PyErr_NewException("deepdrive_client.time_out", NULL, NULL);
		Py_INCREF(TimeOutError);
		PyModule_AddObject(m, "time_out", TimeOutError);

		ClientDoesntExistError = PyErr_NewException("deepdrive_client.client_doesnt_exist", NULL, NULL);
		Py_INCREF(ClientDoesntExistError);
		PyModule_AddObject(m, "client_doesnt_exist", ClientDoesntExistError);

		UnknownError = PyErr_NewException("deepdrive_client.unknown_error", NULL, NULL);
		Py_INCREF(UnknownError);
		PyModule_AddObject(m, "unknown_error", UnknownError);

		Py_INCREF(&PySimulationGraphicsSettingsType);
		PyModule_AddObject(m, "SimulationGraphicsSettings", (PyObject *)&PySimulationGraphicsSettingsType);
/*
*/

		// Py_INCREF(&PyDeepDriveClientRegisterClientRequestType);
		// PyModule_AddObject(m, "RegisterClientRequest", (PyObject *)&PyDeepDriveClientRegisterClientRequestType);
	}
	std::cout << "### ><>|><> PyInit_deepdrive_client <><|<>< ###\n";

	return m;
}
