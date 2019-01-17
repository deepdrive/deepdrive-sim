import asyncio
import collections
import time
import traceback
import re
from itertools import chain

try:
    import unreal_engine as ue
except ImportError:
    print('Cannot import unreal engine')

import pyarrow


KPH_2_MPH = 1. / 0.6213711922


class LambdaApi(object):
    def __init__(self):
        self.sim = None
        self.world = None
        self.ue_objects_by_type = None

    def get_objects_of_type(self, object_type, world):
        if self.ue_objects_by_type is None:
            self.populate_objects_by_type(world)
        return self.ue_objects_by_type[object_type]

    def populate_objects_by_type(self, world):
        ret = collections.defaultdict(list)
        objects = world.all_objects()
        for o in objects:
            class_name = o.get_class().get_name()
            # ueprint('str-class', str(class_name))
            ret[class_name].append(o)
        # print('\n'.join(sorted(objs.keys())))
        self.ue_objects_by_type = ret
        return ret

    def get_sim(self, world=None):
        # TODO: Add sim and world to a singleton
        self.world = world or self.get_world()
        sim_objs  = self.get_objects_of_type('DeepDriveSim_C', self.world)
        if len(sim_objs) > 1:
            raise ValueError('Sim is a singleton, should be only one, but got %r' % len(sim_objs))
        elif not sim_objs:
            print('Could not find DeepDriveSim object')
        self.sim = sim_objs[0]
        return self.sim

    def get_actor_by_name(self, name, world=None):
        self.world = world or self.get_world()
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

    def reset(self, enable_traffic=False):
        sim = self.get_sim()
        if not enable_traffic:
            print('Disabling traffic for next reset...')
        sim.ResetSimulation(enable_traffic)

    def set_ego_kph(self, min_kph, max_kph):
        ego = self.get_ego_agent()
        return ego.SetSpeedRange(min_kph, max_kph)

    def set_ego_mph(self, min_mph, max_mph):
        return self.set_ego_kph(min_mph * KPH_2_MPH, max_mph * KPH_2_MPH)

    def get_ego_agent(self):
        sim = self.get_sim()
        agents = sim.GetAgentsList()
        ego = [a for a in agents if a.IsEgoAgent()]
        if ego:
            return ego[0]
        else:
            return None

    def get_42(self):
        """For sanity testing server outside of Unreal"""
        return 42

    def get_world(self):
        if self.world is not None:
            return self.world
        worlds = ue.all_worlds()
        # print('All worlds length ' + str(len(worlds)))

        # print([w.get_full_name() for w in worlds])

        if hasattr(ue, 'get_editor_world'):
            print('Detected Unreal Editor')
            worlds.append(ue.get_editor_world())

        worlds = [w for w in worlds if 'DeepDriveSim_Demo.DeepDriveSim_Demo' in w.get_full_name()]

        for w in worlds:
            ai_controllers = [a for a in w.all_actors()
                              if 'LocalAIController_'.lower() in a.get_full_name().lower()]
            if ai_controllers:
                print('Found current world! ' + str(w))
                self.world = w
                return w
        else:
            print('Current world not detected')
            return None


def ueprint(*args, **kwargs):
    args += tuple(chain.from_iterable(kwargs.items()))
    print(' '.join(str(x) for x in args))


if __name__ == '__main__':
    pass
    # disable_traffic_next_reset()
    # set_ego_mph(30, 30)
