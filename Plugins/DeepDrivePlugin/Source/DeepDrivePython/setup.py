from __future__ import print_function
import os
from setuptools import setup, Extension, find_packages
from sys import platform
import numpy as np
from datetime import datetime

SRC_DIR = os.environ.get('DEEPDRIVE_SRC_DIR', '..')


sources_capture =	[	SRC_DIR + '/DeepDrivePlugin/Private/SharedMemory/SharedMemory.cpp'
                    ,	'DeepDriveSharedMemoryClient.cpp'
                    ,	'deepdrive.cpp'
                    ,	'NumPyUtils.cpp'
                    ]

sources_control =	[	SRC_DIR + '/DeepDrivePlugin/Private/SharedMemory/SharedMemory.cpp'
                    ,	'DeepDriveControl.cpp'
                    ,	'deepdrive_control.cpp'
                    ,	'NumPyUtils.cpp'
                    ]

includes =	[	np.get_include()
            ,	'include/Unreal'
            ,	SRC_DIR + '/DeepDrivePlugin'
            ]

macros = []
compiler_args = []

if platform == "linux" or platform == "linux2":
    macros.append(('DEEPDRIVE_PLATFORM_LINUX', None))
    sources_capture.append(SRC_DIR + '/DeepDrivePlugin/Private/SharedMemory/SharedMemoryImpl_Linux.cpp')
    sources_control.append(SRC_DIR + '/DeepDrivePlugin/Private/SharedMemory/SharedMemoryImpl_Linux.cpp')
    compiler_args.append('-std=c++11')
elif platform == "darwin":
    # MacOs
    macros.append(('DEEPDRIVE_PLATFORM_MAC', None))
    raise Exception('Mac not supported - shared memory implementation still needed')
elif platform == "win32":
    macros.append(('DEEPDRIVE_PLATFORM_WINDOWS', None))
    sources_capture.append(SRC_DIR + '/DeepDrivePlugin/Private/SharedMemory/SharedMemoryImpl_Windows.cpp')
    sources_control.append(SRC_DIR + '/DeepDrivePlugin/Private/SharedMemory/SharedMemoryImpl_Windows.cpp')
    print('Detected Windows platform')

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

setup	(   name='deepdrive'
        ,   version=os.environ['DEEPDRIVE_VERSION']
        ,   url='https://github.com/deepdrive/deepdrive-unreal'
        ,   author='deepdrive.io'
        ,   author_email='developers@deepdrive.io'
        ,   license='MIT'
        ,   description='Python interface to vehicle simulation running in Unreal'
        ,   ext_modules=[deepdrive_capture_module, deepdrive_control_module]
        ,   include_dirs=includes
        ,   install_requires=['numpy']
        )


print(np.get_include())
