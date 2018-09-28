"""
On Editor/Engine start, the ue_site module is tried for import. You should place initialization code there.
If the module cannot be imported, you will get a (harmful) message in the logs.

This file is also necessary for packaged builds
"""
from pathlib import Path

try:
    import zmq
    import pyarrow
    START_UNREAL_API_SERVER = True
    print('Starting UnrealPython server on zmq!')
except ImportError:
    START_UNREAL_API_SERVER = False
    CURR_PATH = Path(__file__).resolve().parent
    print('To enable the UnrealPython API, start the sim '
          '\n\tthrough the deepdrive project (github.com/deepdrive/deepdrive)'
          '\n\tand enter %s'
          '\n\tas the simulator project directory when prompted.'
          '\n\tAlternately, you can download the dependencies from'
          '\n\thttps://s3-us-west-1.amazonaws.com/deepdrive/unreal_python_lib/python_libs.zip'
          '\n\tand extract into <your-project-root>/python_libs' % CURR_PATH)


def start_unreal_api_server():
    pass


if START_UNREAL_API_SERVER:
    start_unreal_api_server()
