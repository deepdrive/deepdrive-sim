import unreal_engine as ue
from pathlib import Path
import asyncio
import sys

from lambda_server import LambdaServer

IS_LINUX = sys.platform == 'linux' or sys.platform == 'linux2'
IS_WINDOWS = sys.platform == 'win32'

"""Liason class between Python and Unreal that allows hooking into the world tick"""

print('Loading DummyPyActor')

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
          '\n\tas the simulator project directory when prompted.')

    if IS_LINUX:
        print('\n\tAlternatively, you can download the dependencies from'
              '\n\thttps://s3-us-west-1.amazonaws.com/deepdrive/unreal_python_lib/python_libs.zip'
              '\n\tand extract into <your-project-root>/python_libs' % CURR_PATH)
    elif IS_WINDOWS:
        print('\n\tAlternatively, you can download the dependencies from'
              '\n\thttps://s3-us-west-1.amazonaws.com/deepdrive/unreal_python_lib/Win64Python35.zip'
              '\n\tand extract into <your-project-root>/Binaries' % CURR_PATH)


class DummyPyActor:

    def __init__(self):
        self.worlds = None
        self.world = None
        self.ai_controllers = None
        self.started_server = False
        self.event_loop = None
        self.lambda_server = None

    def begin_play(self):
        print('Begin Play on DummyPyActor class')
        self._find_world()

    def tick(self, delta_time):
        if self.world is None:
            self._find_world()
        elif not self.started_server:
            print('Creating new event loop. You should only see this once!')
            self.event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.event_loop)
            # self.event_loop.set_debug(enabled=True)
            print('Creating server task')
            self.lambda_server = LambdaServer()
            asyncio.ensure_future(self.lambda_server.run(self.world))
            self.started_server = True
        elif self.event_loop is not None:
            self.event_loop.stop()
            self.event_loop.run_forever()

    def end_play(self, end_code):
        self.lambda_server.close()
        print('Closing lambda server event loop')
        self.event_loop.close()

    def _find_world(self):
        self.worlds = ue.all_worlds()
        # print('All worlds length ' + str(len(self.worlds)))

        # print([w.get_full_name() for w in self.worlds])

        if hasattr(ue, 'get_editor_world'):
            # print('Detected Unreal Editor')
            self.worlds.append(ue.get_editor_world())
        else:
            # print('Determined we are in a packaged game')
            # self.worlds.append(self.uobject.get_current_level()) # A LEVEL IS NOT A WORLD
            pass

        self.worlds = [w for w in self.worlds if 'DeepDriveSim_Demo.DeepDriveSim_Demo' in w.get_full_name()]

        for w in self.worlds:
            self.ai_controllers = [a for a in w.all_actors()
                                   if 'LocalAIController_'.lower() in a.get_full_name().lower()]
            if self.ai_controllers:
                self.world = w
                print('Found current world: ' + str(w))
                break
        else:
            # print('Current world not detected')
            pass

        return self.world


