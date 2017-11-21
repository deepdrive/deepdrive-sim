from distutils.core import setup, Extension
import numpy as np
from sys import platform

sources_capture =	[	'../DeepDrivePlugin/Private/SharedMemory/SharedMemory.cpp'
					,	'DeepDriveSharedMemoryClient.cpp'
					,	'deepdrive.cpp'
					,	'NumPyUtils.cpp'
					]

sources_control =	[	'../DeepDrivePlugin/Private/SharedMemory/SharedMemory.cpp'
					,	'DeepDriveControl.cpp'
					,	'deepdrive_control.cpp'
					,	'NumPyUtils.cpp'
					]

includes =	[	np.get_include()
			,	'include/Unreal'
			,	'../DeepDrivePlugin'
			]

macros = []
compiler_args = []

if platform == "linux" or platform == "linux2":
	macros.append(('DEEPDRIVE_PLATFORM_LINUX', None))
	sources_capture.append('../DeepDrivePlugin/Private/SharedMemory/SharedMemoryImpl_Linux.cpp')
	sources_control.append('../DeepDrivePlugin/Private/SharedMemory/SharedMemoryImpl_Linux.cpp')
	compiler_args.append('-std=c++11')
elif platform == "darwin":
	# MacOs - we'll just hope shared memory works for now
	macros.append(('DEEPDRIVE_PLATFORM_MAC', None))
elif platform == "win32":
	macros.append(('DEEPDRIVE_PLATFORM_WINDOWS', None))
	sources_capture.append('../DeepDrivePlugin/Private/SharedMemory/SharedMemoryImpl_Windows.cpp')
	sources_control.append('../DeepDrivePlugin/Private/SharedMemory/SharedMemoryImpl_Windows.cpp')
	print( 'Detected Windows platform')

deepdrive_capture_module = Extension	(	'deepdrive'
										,	sources=sources_capture
										,	extra_compile_args=compiler_args
										,	define_macros=macros
										)

deepdrive_control_module = Extension	(	'deepdrive_control'
										,	sources=sources_control
										,	extra_compile_args=compiler_args
										,	define_macros=macros
										)

setup	(	name='deepdrive'
		,	version='0.0.1'
		,	description='Python interface to vehicle simulation running in Unreal'
		,	ext_modules=[deepdrive_capture_module, deepdrive_control_module]
		,	include_dirs=includes
		,	install_requires=['numpy']
		)


print(np.get_include())
