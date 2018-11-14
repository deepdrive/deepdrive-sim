import asyncio
import time
import traceback
import re

try:
    import unreal_engine as ue
except ImportError:
    print('Cannot import unreal engine')

import pyarrow

def get_actor_by_name(name, world):
    actors = [(a.get_full_name(), a) for a in world.all_actors()]
    print('num actors = ' + str(len(actors)))
    pattern = r'.*\.' + name + '_.*\d+'
    r = re.compile(pattern)
    ret = None
    for a_name, a in actors:
        if r.match(a_name):
            print(a_name)
            print(a)
            print(a.HasAdditionalAgents)
            a.HasAdditionalAgents = True
            # for d in dir(a):
            #     print(d)
            # return a
            ret = a
    return ret


def get_world():
    worlds = ue.all_worlds()
    # print('All worlds length ' + str(len(worlds)))

    # print([w.get_full_name() for w in worlds])

    if hasattr(ue, 'get_editor_world'):
        # print('Detected Unreal Editor')
        worlds.append(ue.get_editor_world())
    else:
        # print('Determined we are in a packaged game')
        # worlds.append(self.uobject.get_current_level()) # A LEVEL IS NOT A WORLD
        pass

    worlds = [w for w in worlds if 'DeepDriveSim_Demo.DeepDriveSim_Demo' in w.get_full_name()]

    for w in worlds:
        ai_controllers = [a for a in w.all_actors()
                               if 'LocalAIController_'.lower() in a.get_full_name().lower()]
        if ai_controllers:
            world = w
            print('Found current world: ' + str(w))
            break
    else:
        # print('Current world not detected')
        pass

    return world


if __name__ == '__main__':
    print(get_actor_by_name('DeepDriveInputController', get_world()))

