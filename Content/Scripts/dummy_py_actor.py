import unreal_engine as ue
from pathlib import Path
import asyncio
import sys
import importlib

import api_server
import api_methods

# Reload modules so we don't have to restart Unreal Editor after changing them
importlib.reload(api_server)
importlib.reload(api_methods)

from api_server import ApiServer

IS_LINUX = sys.platform == 'linux' or sys.platform == 'linux2'
IS_WINDOWS = sys.platform == 'win32'


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
              '\n\thttps://s3-us-west-1.amazonaws.com/deepdrive/embedded_python_for_unreal/linux/python_libs.zip'
              '\n\tand extract into <your-project-root>/python_libs' % CURR_PATH)
    elif IS_WINDOWS:
        print('\n\tAlternatively, you can download the dependencies from'
              '\n\thttps://s3-us-west-1.amazonaws.com/deepdrive/embedded_python_for_unreal/windows/python_bin_with_libs.zip'
              '\n\tand extract into <your-project-root>/Binaries/Win64' % CURR_PATH)


class DummyPyActor:
    """
    Liaison class between Python and Unreal that allows hooking into the
    world tick
    """
    def __init__(self):
        self.worlds = None
        self.world = None
        self.ai_controllers = None
        self.started_server = False
        self.event_loop = None
        self.api_server = None

    def begin_play(self):
        print('Begin Play on DummyPyActor class')
        self._find_world()

    def tick(self, delta_time):
        if self.world is None:
            print('Searching for unreal world for UnrealEnginePython')
            self._find_world()
        elif not self.started_server:
            print('Creating new event loop. You should only see this once!')
            self.event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.event_loop)
            self.event_loop.set_debug(enabled=True)
            print('Creating server task')
            self.api_server = ApiServer()
            asyncio.ensure_future(self.api_server.run())
            self.started_server = True
        elif self.event_loop is not None:
            self.event_loop.stop()
            self.event_loop.run_forever()

    def end_play(self, end_code):
        if self.event_loop is not None:
            self.event_loop.stop()
        self.api_server.close()
        print('Closing api server event loop')
        self.event_loop.close()

    def _find_world(self):
        # TODO: Use api_methods.find_world
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

        self.worlds = [w for w in self.worlds if valid_world(w)]
        print(self.worlds)

        for w in self.worlds:
            # [print(a.get_full_name()) for a in w.all_actors()]
            self.ai_controllers = [a for a in w.all_actors()
                                   if valid_ai_controller_name(a)]
            if self.ai_controllers:
                self.world = w
                print('Found current world: ' + str(w))
                break
            else:
                print(f'no valid ai controller found in {w.get_full_name()}')
        else:
            # print('Current world not detected')
            pass

        return self.world


def valid_world(world):
    full_name = world.get_full_name()
    substrings = [
        'DeepDriveSim_Demo.DeepDriveSim_Demo',
        'DeepDriveSim_Kevindale_Full.DeepDriveSim_Kevindale_Full',
        'DeepDriveSim_Kevindale_Bare.DeepDriveSim_Kevindale_Bare',
        'DeepDriveSim_CityMap_Demo.DeepDriveSim_CityMap_Demo',
        'DeepDriveSim_CityMapTraffic_Demo.DeepDriveSim_CityMapTraffic_Demo',
    ]
    for s in substrings:
        if s in full_name:
            return True
    return False

def valid_ai_controller_name(actor):
    full_name = actor.get_full_name().lower()
    substrings = ['.LocalAIControllerCreator_'.lower()]
    for s in substrings:
        if s in full_name:
            return True
    return False