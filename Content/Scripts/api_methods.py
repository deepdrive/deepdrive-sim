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

KPH_2_MPH = 0.6213711922
MPH_2_KPH = 1. / KPH_2_MPH
SUPPORTED_TYPES = [int, float, str, bool]
SUPPORTED_UOBJECT_PREFIXES = ["<unreal_engine.UObject 'DeepDriveAgent_"]


def get_standard_actor_props(actor):
    # noinspection PyDictCreation
    out = {}
    out['position'] = best_effort_serialize(actor.get_actor_location())
    # out['right_vector'] = actor.get_right_vector()
    # out['physics_linear_velocity'] = actor.get_physics_linear_velocity()
    # out['physics_angular_velocity'] = actor.get_physics_angular_velocity()
    out['actor_bounds'] = best_effort_serialize(actor.get_actor_bounds())
    out['actor_right'] = best_effort_serialize(actor.get_actor_right())
    out['actor_rotation'] = best_effort_serialize(actor.get_actor_rotation())
    out['actor_scale'] = best_effort_serialize(actor.get_actor_scale())
    out['actor_transform'] = best_effort_serialize(actor.get_actor_transform())
    out['actor_up'] = best_effort_serialize(actor.get_actor_up())
    out['actor_velocity'] = best_effort_serialize(actor.get_actor_velocity())
    return out


class Api(object):
    def __init__(self):
        self._world = None
        self._sim = None
        self.ue_objects_by_type = None

    @property
    def sim(self):
        if self._sim is None:
            self._sim = self.get_sim()
        return self._sim
    
    @property
    def world(self):
        if self._world is None:
            self._world = self.get_world()
        return self._world

    def get_observation(self):
        # Be careful with what you add here as this is much slower than
        # our shared memory channel for sending observations.
        ret = dict(vehicle_positions=self.get_vehicle_positions())
        return ret

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
        world = world or self.world
        sim_objs = self.get_objects_of_type('DeepDriveSim_C', world)
        if len(sim_objs) > 1:
            raise ValueError('Sim is a singleton, should be only one,'
                             ' but got %r' % len(sim_objs))
        elif not sim_objs:
            print('Could not find DeepDriveSim object')
        ret = sim_objs[0]
        return ret

    def get_actor_by_name(self, name, world=None):
        world = world or self.get_world()
        pattern = name + r'(_.*\d+)?$'
        r = re.compile(pattern)
        actors = [a for a in world.all_actors() if a.get_display_name() == name
                  or r.match(a.get_display_name())]
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
        return self.set_ego_kph(min_mph * MPH_2_KPH, max_mph * MPH_2_KPH)

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

    def get_agents(self):
        ret = []
        agents = self.sim.GetAgentsList()
        # print('\n'.join(dir(agents[0])))
        # print(str(agents[0]))
        for a in agents:
            ra = {}
            levels = 3
            ura = best_effort_serialize(a, levels)
            ra['reflection_%r_level' % levels] = ura
            ra['is_ego'] = a.IsEgoAgent()
            ra['actor_info'] = get_standard_actor_props(a)
            ret.append(ra)
        return ret

    def get_agent_positions(self):
        ret = []
        agents = self.sim.GetAgentsList()

        # print([dir(a) for a in agents])
        for agent in agents:
            if not agent.IsEgoAgent():
                position = agent.get_actor_location()
                ret.append([position.x, position.y, position.z])
        return ret

    def get_world(self):
        worlds = ue.all_worlds()
        # print('All worlds length ' + str(len(worlds)))

        # print([w.get_full_name() for w in worlds])

        if hasattr(ue, 'get_editor_world'):
            print('Detected Unreal Editor')
            worlds.append(ue.get_editor_world())

        # worlds = [w for w in worlds if 'DeepDriveSim_Demo.DeepDriveSim_Demo' in
        #           w.get_full_name()]

        # print([w.get_full_name() for w in worlds])

        for w in worlds:
            ai_controllers = [a for a in w.all_actors()
                              if 'AIController_'.lower()
                              in a.get_full_name().lower()]
            if ai_controllers:
                print('Found current world! ' + str(w))
                return w
        else:
            print('Current world not detected')
            return None


def get_rotator(v):
    ret = dict(roll=v.roll, pitch=v.pitch, yaw=v.yaw)
    return ret


def get_transform(v):
    ret = dict(
        translation=best_effort_serialize(v.translation),
        rotation=best_effort_serialize(v.rotation),
        scale=best_effort_serialize(v.scale))
    return ret


def best_effort_serialize(v, levels=2):
    if any(isinstance(v, stype) for stype in SUPPORTED_TYPES):
        new = v
    elif str(type(v)) == "<class 'unreal_engine.FVector'>":
        new = get_3d_vector(v)
    elif str(type(v)) == "<class 'unreal_engine.FRotator'>":
        new = get_rotator(v)
    elif str(type(v)) == "<class 'unreal_engine.FTransform'>":
        new = get_transform(v)
    elif levels == 0:
        new = str(v)
    elif type(v) is list or type(v) is set or type(v) is tuple:
        print('serializing list')
        new = [best_effort_serialize(x, levels-1) for x in v]
    elif type(v) is dict:
        print('serializing dict')
        new = {k2: best_effort_serialize(v2, levels-1) for k2, v2 in v.items()}
    else:
        print('serializing unknown')
        if not any(str(v).startswith(st) for st in SUPPORTED_UOBJECT_PREFIXES):
            new = str(v)
        else:
            try:
                new = best_effort_serialize(v.as_dict(), levels-1)
            except Exception as e0:
                try:
                    print('Can\'t serialize ' + str(v) +
                          ' converting to string, error: ' + str(e0))
                    new = str(v)
                except Exception as e1:
                    new = 'Can\'t serialize, error: ' + str(e1)

    return new


def ueprint(*args, **kwargs):
    args += tuple(chain.from_iterable(kwargs.items()))
    print(' '.join(str(x) for x in args))


def get_3d_vector(p):
    return [p.x, p.y, p.z]


def main():
    api = Api()
    agents = api.get_agents()
    import json
    print(json.dumps(agents, indent=2, sort_keys=True))
    # import pprint
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(agents)

    # disable_traffic_next_reset()
    # set_ego_mph(30, 30)


if __name__ == '__main__':
    main()
