from __future__ import print_function
import os
from setuptools import setup, Extension
from sys import platform
import numpy as np
from build import config

SRC_DIR = os.environ.get('DEEPDRIVE_SRC_DIR', '..')

print('###################################')
print(SRC_DIR + '/DeepDrivePlugin/Private/SharedMemory/SharedMemory.cpp')
print(os.environ['DEEPDRIVE_VERSION'])
print('###################################')

sources_capture = [SRC_DIR + '/DeepDrivePlugin/Private/SharedMemory/SharedMemory.cpp'
    , 'src/deepdrive_capture/DeepDriveSharedMemoryClient.cpp'
    , 'src/deepdrive_capture/deepdrive_capture.cpp'
    , 'src/deepdrive_capture/PyCaptureCameraObject.cpp'
    , 'src/deepdrive_capture/PyCaptureLastCollisionObject.cpp'
    , 'src/deepdrive_capture/PyCaptureSnapshotObject.cpp'
    , 'src/common/NumPyUtils.cpp'
                   ]

sources_client = ['src/deepdrive_client/deepdrive_client.cpp'
    , 'src/deepdrive_client/DeepDriveClient.cpp'
    , 'src/deepdrive_client/DeepDriveClientMap.cpp'
    , 'src/deepdrive_simulation/DeepDriveSimulation.cpp'
    , 'src/socket/IP4Address.cpp'
    , 'src/socket/IP4ClientSocket.cpp'
    , 'src/common/NumPyUtils.cpp'
                  ]

sources_simulation = ['src/deepdrive_simulation/deepdrive_simulation.cpp'
    , 'src/deepdrive_simulation/deepdrive_simulation_multi_agent.cpp'
    , 'src/deepdrive_simulation/deepdrive_simulation_error.cpp'
    , 'src/deepdrive_simulation/DeepDriveSimulation.cpp'
    , 'src/deepdrive_simulation/PyMultiAgentSnapshotObject.cpp'
    , 'src/socket/IP4Address.cpp'
    , 'src/socket/IP4ClientSocket.cpp'
    , 'src/common/NumPyUtils.cpp'
                      ]

includes = ['./'
    , 'src'
    , 'include/Unreal'
    , SRC_DIR + '/DeepDrivePlugin'
    , np.get_include()
            ]

macros = []
compiler_args = []

if platform == "linux" or platform == "linux2":
    macros.append(('DEEPDRIVE_PLATFORM_LINUX', None))
    sources_capture.append(SRC_DIR + '/DeepDrivePlugin/Private/SharedMemory/SharedMemoryImpl_Linux.cpp')
    sources_client.append('src/socket/IP4ClientSocketImpl_Linux.cpp')
    sources_simulation.append('src/socket/IP4ClientSocketImpl_Linux.cpp')
    compiler_args.append('-std=c++11')
elif platform == "darwin":
    # MacOs
    macros.append(('DEEPDRIVE_PLATFORM_MAC', None))
    raise Exception('Mac not supported - shared memory implementation still needed')
elif platform == "win32":
    macros.append(('DEEPDRIVE_PLATFORM_WINDOWS', None))
    sources_capture.append(SRC_DIR + '/DeepDrivePlugin/Private/SharedMemory/SharedMemoryImpl_Windows.cpp')
    sources_client.append('src/socket/IP4ClientSocketImpl_Windows.cpp')
    sources_simulation.append('src/socket/IP4ClientSocketImpl_Windows.cpp')
    print('Detected Windows platform')

deepdrive_capture_module = Extension('deepdrive_capture'
                                     , sources=sources_capture
                                     , extra_compile_args=compiler_args
                                     , define_macros=macros
                                     )

deepdrive_client_module = Extension('deepdrive_client'
                                    , sources=sources_client
                                    , extra_compile_args=compiler_args
                                    , define_macros=macros
                                    )

deepdrive_simulation_module = Extension('deepdrive_simulation'
                                        , sources=sources_simulation
                                        , extra_compile_args=compiler_args
                                        , define_macros=macros
                                        )

setup(name=config.PACKAGE_NAME
      , version=os.environ['DEEPDRIVE_VERSION'] + os.environ.get('DEEPDRIVE_VERSION_SEGMENT', '')
      , url='https://github.com/deepdrive/deepdrive-sim'
      , author='deepdrive.io'
      , author_email='developers@deepdrive.io'
      , license='MIT'
      , description='Python bindings for Deepdrive interface to Unreal'
      , ext_modules=[deepdrive_capture_module, deepdrive_client_module, deepdrive_simulation_module]
      , include_dirs=includes
      , install_requires=['numpy']
      )
