#include "Python.h"

#include "DeepDriveControl.h"
#include "PyDeepDriveControlObject.h"
#include <iostream>

static PyObject *DeepDriveError;

static DeepDriveControl *g_DeepDriveControl = 0;

/*	Reset connection to UE environmnent by trying to create a shared memory connection
 *
 *	@param	string		Name of shared memory
 *	qparam	uint32		Maximiun size of share memory
 *	@return	True, if successfully, otherwise false
*/
static PyObject* deepdrive_control_reset(PyObject *self, PyObject *args)
{
	uint32 res = 0;

	if(g_DeepDriveControl)
		delete g_DeepDriveControl;

	g_DeepDriveControl = new DeepDriveControl();

	if(g_DeepDriveControl)
	{
		const char *sharedMemName = 0;
		uint32 maxSize = 0;

		const int paramsOk = PyArg_ParseTuple(args, "sI", &sharedMemName, &maxSize);

		if(paramsOk)
		{
			std::cout << "Try to open " << sharedMemName << " with size of " << maxSize << "\n";

			res = g_DeepDriveControl->create(sharedMemName, maxSize);

		}
	}
	return Py_BuildValue("i", res);
}


/*	Send disconnect message to UE environmnent
 *
 *
*/
static PyObject* deepdrive_control_disconnect(PyObject *self, PyObject *args)
{
	std::cout << "\ndeepdrive_control_disconnect\n";

	if(g_DeepDriveControl)
	{
		g_DeepDriveControl->disconnect();
	}
	
	return Py_BuildValue("i", 1);
}

/*	Close connection to UE environmnent
 *
 *
*/
static PyObject* deepdrive_control_close(PyObject *self, PyObject *args)
{
//	std::cout << "\ndeepdrive_control_close\n";

	if(g_DeepDriveControl)
	{
		delete g_DeepDriveControl;
		g_DeepDriveControl = 0;
	}
	
	return Py_BuildValue("i", 1);
}

/*	Reset connection to UE environmnent by trying to create a shared memory connection
 *
 *	@param	DeepDrvie		Maximiun size of share memory
 *	@return	True, if successfully, otherwise false
*/
static PyObject* deepdrive_control_send_control(PyObject *self, PyObject *args)
{
	uint32 res = 0;

	if(g_DeepDriveControl)
	{
		const PyDeepDriveControlObject *ctrlObj = 0;
		uint32 maxSize = 0;

		const int paramsOk = PyArg_ParseTuple(args, "O!", &PyDeepDriveControlType, &ctrlObj);

		if(paramsOk)
		{
//			std::cout << "deepdrive_control_send_control \n";
			g_DeepDriveControl->sendControl( *ctrlObj );

		}
	}
	return Py_BuildValue("i", res);
}


static PyMethodDef DeepDriveControlMethods[] =	{	{"reset", deepdrive_control_reset, METH_VARARGS, "Reset environmnent and tries to create a shared memory connection"}
										,	{"disconnect", deepdrive_control_disconnect, METH_VARARGS, "Send disconnect message to UE environmnent"}
										,	{"close", deepdrive_control_close, METH_VARARGS, "Close connection to UE environmnent"}
										,	{"send_control", deepdrive_control_send_control, METH_VARARGS, "Close connection to UE environmnent"}
										,	{NULL,     NULL,             0,            NULL}        /* Sentinel */
										};

static struct PyModuleDef deepdrive_control_module = {
		PyModuleDef_HEAD_INIT,
		"deepdrive_control",   /* name of module */
		NULL,          /* module documentation, may be NULL */
		-1,            /* size of per-interpreter state of the module,
				          or -1 if the module keeps state in global variables. */
		DeepDriveControlMethods
};

PyMODINIT_FUNC PyInit_deepdrive_control(void)
{
	if (PyType_Ready(&PyDeepDriveControlType) < 0)
	return 0;

	PyObject *m  = PyModule_Create(&deepdrive_control_module);
	if (m)
	{
		DeepDriveError = PyErr_NewException("deepdrive_control.error", NULL, NULL);
		Py_INCREF(DeepDriveError);
		PyModule_AddObject(m, "error", DeepDriveError);

		Py_INCREF(&PyDeepDriveControlType);
		PyModule_AddObject(m, "DeepDriveControl", (PyObject *)&PyDeepDriveControlType);
	}

	return m;
}
