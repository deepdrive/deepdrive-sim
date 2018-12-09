import asyncio
import time
import traceback
import re

try:
    import unreal_engine as ue
except ImportError:
    print('Cannot import unreal engine')

import pyarrow

WORLD = None

def get_actor_by_name(name, world=None):
    global WORLD
    if WORLD is None:
        WORLD = get_world()
    world = world or WORLD
    pattern = name + r'(_.*\d+)?$'
    r = re.compile(pattern)
    actors = [a for a in world.all_actors() if a.get_display_name() == name or r.match(a.get_display_name())]
    ret = None
    if len(actors) > 0:
        ret = actors[0]

    if len(actors) > 1:
        print('Found multiple actors matching ' + name + '!')
        for a in actors:
            print('%s %s' % (a.get_display_name(), a.get_full_name))

    return ret


def get_world():
    global WORLD
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
            WORLD = w
            print('Found current world: ' + str(w))
            break
    else:
        # print('Current world not detected')
        pass

    return WORLD


def enable_traffic_next_reset():
    input_controller = get_actor_by_name('DeepDriveInputController')
    print('Enabling traffic for next reset...')
    if input_controller.HasAdditionalAgents:
        print('Traffic already enabled for next reset')
    else:
        print('Traffic will be enabled on next reset')
        input_controller.HasAdditionalAgents = True



def disable_traffic_next_reset():
    input_controller = get_actor_by_name('DeepDriveInputController')
    print('Disabling traffic for next reset...')
    if not input_controller.HasAdditionalAgents:
        print('Traffic already disabled for next reset')
    else:
        print('Traffic will be disabled on next reset')
        input_controller.HasAdditionalAgents = False


if __name__ == '__main__':
    # disable_traffic_next_reset()
    pass
