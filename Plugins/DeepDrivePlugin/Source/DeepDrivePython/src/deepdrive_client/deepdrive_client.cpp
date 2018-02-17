#include "Python.h"

#include "Engine.h"

#include "deepdrive_client/DeepDriveClient.hpp"
#include "common/ClientErrorCode.hpp"

#include "common/NumPyUtils.h"

#include "numpy/arrayobject.h"

#include <iostream>
#include <map>

typedef std::map<uint32, DeepDriveClient*>		ClientMap;

static PyObject *DeepDriveClientError;
static PyObject *ConnectionLostError;
static PyObject *NotConnectedError;
static PyObject *TimeOutError;
static PyObject *ClientDoesntExistError;

static ClientMap g_Clients;

static DeepDriveClient* getClient(uint32 clientId)
{
	ClientMap::iterator cIt = g_Clients.find(clientId);
	DeepDriveClient *client = cIt != g_Clients.end() ? cIt->second : 0;
	return client;
}

/*	Create a new client, tries to connect to specified DeepDriveServer
 *
 *	@param	address		IP4 address of server
 *	@param	uint32		Request master role
 *
 *	@return	Client id, 0 in case of error
*/
static PyObject* deepdrive_client_create(PyObject *self, PyObject *args)
{
	deepdrive::server::RegisterClientResponse res;

	uint32 clientId = 0;
	PyObject *ret = PyDict_New();

	const char *ipStr;
	uint32 port = 19768;
	int32 ok = PyArg_ParseTuple(args, "s|i", &ipStr, &port);

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
				const int32 res = client->registerClient(registerClientResponse);
				if(res >= 0)
				{
					clientId = registerClientResponse.client_id;
					std::cout << "Client id is " << std::to_string(clientId) << "\n";
					PyDict_SetItem(ret, PyUnicode_FromString("client_id"), PyLong_FromUnsignedLong(clientId));
					if(clientId)
					{
						g_Clients[clientId] = client;
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
				else if(res == ClientErrorCode::CONNECTION_LOST)
				{
					PyErr_SetString(ClientDoesntExistError, "Client doesn't exist");
					return 0;
				}
				else if(res == ClientErrorCode::TIME_OUT)
				{
					PyErr_SetString(TimeOutError, "Network time out");
					return 0;
				}
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
		ClientMap::iterator cIt = g_Clients.find(clientId);
		if(cIt != g_Clients.end())
		{
			DeepDriveClient *client = cIt->second;
			if(client)
			{
				client->close();
				delete client;
			}
			g_Clients.erase(cIt);
		}
		else
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
				{	
					if(res == ClientErrorCode::CONNECTION_LOST)
					{
						PyErr_SetString(ConnectionLostError, "Connection to server lost");
						return 0;
					}
					else if(res == ClientErrorCode::NOT_CONNECTED)
					{
						PyErr_SetString(NotConnectedError, "Not connected to server");
						return 0;
					}
					else
						res = 0;
				}
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

/*	Request agent control
 *
 *	@param	uint32		Client Id
 *
 *	@return	True, if conmtrol was granted, otherwise false
*/
static PyObject* deepdrive_client_request_agent_control(PyObject *self, PyObject *args)
{
	uint32 clientId = 19768;
	int32 ok = PyArg_ParseTuple(args, "i", &clientId);

	int32 res = 0;

	if(ok && clientId > 0)
	{
		DeepDriveClient *client = getClient(clientId);
		if(client)
		{
			int32 res = client->requestAgentControl();
			if(res < 0)
			{
				if(res == ClientErrorCode::CONNECTION_LOST)
				{
					PyErr_SetString(ConnectionLostError, "Connection to server lost");
					return 0;
				}
				else if(res == ClientErrorCode::NOT_CONNECTED)
				{
					PyErr_SetString(NotConnectedError, "Not connected to server");
					return 0;
				}
				else
					res = 0;
			}
		}
		else
		{
			PyErr_SetString(ClientDoesntExistError, "Client doesn't exist");
			return 0;
		}
	}

	return Py_BuildValue("i", res);
}


/*	Release agent control
 *
 *	@param	uint32		Client Id
 *
*/
static PyObject* deepdrive_client_release_agent_control(PyObject *self, PyObject *args)
{
	uint32 clientId = 19768;
	int32 ok = PyArg_ParseTuple(args, "i", &clientId);

	if(ok && clientId > 0)
	{
		DeepDriveClient *client = getClient(clientId);
		if(client)
		{
			const int32 res = client->releaseAgentControl();
			if(res == ClientErrorCode::CONNECTION_LOST)
			{
				PyErr_SetString(ConnectionLostError, "Connection to server lost");
				return 0;
			}
			else if(res == ClientErrorCode::NOT_CONNECTED)
			{
				PyErr_SetString(NotConnectedError, "Not connected to server");
				return 0;
			}
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
			if(res == ClientErrorCode::CONNECTION_LOST)
			{
				PyErr_SetString(ConnectionLostError, "Connection to server lost");
				return 0;
			}
			else if(res == ClientErrorCode::NOT_CONNECTED)
			{
				PyErr_SetString(NotConnectedError, "Not connected to server");
				return 0;
			}
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
			if(res == ClientErrorCode::CONNECTION_LOST)
			{
				PyErr_SetString(ConnectionLostError, "Connection to server lost");
				return 0;
			}
			else if(res == ClientErrorCode::NOT_CONNECTED)
			{
				PyErr_SetString(NotConnectedError, "Not connected to server");
				return 0;
			}
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
		ClientMap::iterator cIt = g_Clients.find(clientId);
		if(cIt != g_Clients.end())
		{
			DeepDriveClient *client = cIt->second;
			if(client)
			{
				return Py_BuildValue("(si)", client->getSharedMemoryName(), client->getSharedMemorySize());
			}
		}
	}


	return Py_BuildValue("");
}


static PyMethodDef DeepDriveClientMethods[] =	{	{"create", deepdrive_client_create, METH_VARARGS, "Creates a new client which tries to connect to DeepDriveServer"}
												,	{"close", deepdrive_client_close, METH_VARARGS, "Closes an existing client connection and frees all depending resources"}
												,	{"register_camera", (PyCFunction) deepdrive_client_register_camera, METH_VARARGS | METH_KEYWORDS, "Register a capture camera"}
												,	{"get_shared_memory", deepdrive_client_get_shared_memory, METH_VARARGS, "Get shared memory name and size for client"}
												,	{"request_agent_control", deepdrive_client_request_agent_control, METH_VARARGS, "Request control over agent"}
												,	{"release_agent_control", deepdrive_client_release_agent_control, METH_VARARGS, "Release control over agent"}
												,	{"reset_agent", deepdrive_client_reset_agent, METH_VARARGS, "Reset the agent"}
												,	{"set_control_values", (PyCFunction) deepdrive_client_set_control_values, METH_VARARGS | METH_KEYWORDS, "Send control value set to server"}
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
	// if (PyType_Ready(&PyDeepDriveClientRegisterClientRequestType) < 0)
	//	return 0;

	std::cout << "###### ><> >< PyInit_deepdrive_client >< <>< ######\n";

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


		// Py_INCREF(&PyDeepDriveClientRegisterClientRequestType);
		// PyModule_AddObject(m, "RegisterClientRequest", (PyObject *)&PyDeepDriveClientRegisterClientRequestType);
	}

	return m;
}
