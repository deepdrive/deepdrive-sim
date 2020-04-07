
#include "deepdrive_simulation/deepdrive_simulation_error.h"
#include "common/ClientErrorCode.hpp"

static PyObject *DeepDriveClientError = 0;
static PyObject *ConnectionLostError = 0;
static PyObject *NotConnectedError = 0;
static PyObject *TimeOutError = 0;
static PyObject *ClientDoesntExistError = 0;
static PyObject *UnknownError = 0;

void setupErrorTypes(PyObject *module)
{
	DeepDriveClientError = PyErr_NewException("deepdrive_simulation.error", NULL, NULL);
	Py_INCREF(DeepDriveClientError);
	PyModule_AddObject(module, "error", DeepDriveClientError);

	ConnectionLostError = PyErr_NewException("deepdrive_simulation.connection_lost", NULL, NULL);
	Py_INCREF(ConnectionLostError);
	PyModule_AddObject(module, "connection_lost", ConnectionLostError);

	NotConnectedError = PyErr_NewException("deepdrive_simulation.not_connected", NULL, NULL);
	Py_INCREF(NotConnectedError);
	PyModule_AddObject(module, "not_connected", NotConnectedError);

	TimeOutError = PyErr_NewException("deepdrive_simulation.time_out", NULL, NULL);
	Py_INCREF(TimeOutError);
	PyModule_AddObject(module, "time_out", TimeOutError);

	ClientDoesntExistError = PyErr_NewException("deepdrive_simulation.client_doesnt_exist", NULL, NULL);
	Py_INCREF(ClientDoesntExistError);
	PyModule_AddObject(module, "client_doesnt_exist", ClientDoesntExistError);

	UnknownError = PyErr_NewException("deepdrive_simulation.unknown_error", NULL, NULL);
	Py_INCREF(UnknownError);
	PyModule_AddObject(module, "unknown_error", UnknownError);
}

PyObject *handleError(int32 errorCode)
{
	if (errorCode == ClientErrorCode::NOT_CONNECTED)
	{
		PyErr_SetString(ClientDoesntExistError, "Client doesn't exist");
	}
	else if (errorCode == ClientErrorCode::CONNECTION_LOST)
	{
		PyErr_SetString(ConnectionLostError, "Connection to server lost");
	}
	else if (errorCode == ClientErrorCode::TIME_OUT)
	{
		PyErr_SetString(TimeOutError, "Network time out");
	}
	else if (errorCode == ClientErrorCode::UNKNOWN_ERROR)
	{
		PyErr_SetString(UnknownError, "Unknown network error");
	}
	return 0;
}

