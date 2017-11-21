#include "Python.h"

#include "DeepDriveSharedMemoryClient.h"

#include "PyCaptureCameraObject.h"
#include "PyCaptureSnapshotObject.h"

#include <iostream>

static PyObject *DeepDriveError;

static DeepDriveSharedMemoryClient *g_SharedMemClient = 0;

/*	Reset connection to UE environmnent by trying to open a connection to shared memory
 *
 *	@param	string		Name of shared memory
 *	qparam	uint32		Maximiun size of share memory
 *	@return	True, if successfully, otherwise false
*/
static PyObject* deepdrive_reset(PyObject *self, PyObject *args)
{
	uint32 res = 0;

	if(g_SharedMemClient)
		delete g_SharedMemClient;

	g_SharedMemClient = new DeepDriveSharedMemoryClient();

	if(g_SharedMemClient)
	{
		const char *sharedMemName = 0;
		uint32 maxSize = 0;

		const int paramsOk = PyArg_ParseTuple(args, "sI", &sharedMemName, &maxSize);

		if(paramsOk)
		{
//			std::cout << "Try to connect to " << sharedMemName << " with size of " << maxSize << "\n";

			res = g_SharedMemClient->connect(sharedMemName, maxSize);

		}
	}

	return Py_BuildValue("i", res);
}

/*	Query next step from UE environment
 *
 *
 *	@return
*/
static PyObject* deepdrive_step(PyObject *self, PyObject *args)
{
	PyObject *res = g_SharedMemClient ? reinterpret_cast<PyObject*> (g_SharedMemClient->readMessage()) : 0;

	if(res == 0)
	{
	    Py_INCREF(Py_None);
    	res = Py_None;
	}
	return res;
}

/*	Close connection to UE environmnent
 *
 *
*/
static PyObject* deepdrive_close(PyObject *self, PyObject *args)
{
	if(g_SharedMemClient)
	{
		delete g_SharedMemClient;
		g_SharedMemClient = 0;
	}
	
//	std::cout << "\ndeepdrive_close\n";

	return Py_BuildValue("i", 1);
}

static PyMethodDef DeepDriveMethods[] =	{	{"reset", deepdrive_reset, METH_VARARGS, "Reset environmnent and tries to open a connection to shared memory"}
										,	{"step", deepdrive_step, METH_VARARGS, "Query next step from UE environment"}
										,	{"close", deepdrive_close, METH_VARARGS, "Close connection to UE environmnent"}
										,	{NULL,     NULL,             0,            NULL}        /* Sentinel */
										};

static struct PyModuleDef deepdrive_module = {
		PyModuleDef_HEAD_INIT,
		"deepdrive",   /* name of module */
		NULL,          /* module documentation, may be NULL */
		-1,            /* size of per-interpreter state of the module,
				          or -1 if the module keeps state in global variables. */
		DeepDriveMethods
};

PyMODINIT_FUNC PyInit_deepdrive(void)
{
	if (PyType_Ready(&PyCaptureCameraType) < 0)
		return 0;

	if (PyType_Ready(&PyCaptureSnapshotType) < 0)
		return 0;

	PyObject *m  = PyModule_Create(&deepdrive_module);
	if (m)
	{
		DeepDriveError = PyErr_NewException("deepdrive.error", NULL, NULL);
		Py_INCREF(DeepDriveError);
		PyModule_AddObject(m, "error", DeepDriveError);

		Py_INCREF(&PyCaptureCameraType);
		PyModule_AddObject(m, "CaptureCamera", (PyObject *)&PyCaptureCameraType);

		Py_INCREF(&PyCaptureSnapshotType);
		PyModule_AddObject(m, "CaptureSnapshot", (PyObject *)&PyCaptureSnapshotType);

	}

	return m;
}
