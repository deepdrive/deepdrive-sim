import collections

try:
    import unreal_engine as ue
    from unreal_engine.classes import Blueprint
    from unreal_engine import FVector, FRotator
    DRY_RUN = False
    WORLD = ue.get_editor_world()
except ImportError:
    DRY_RUN = True

from api_methods import get_world, ueprint
import api_methods



# TODO: Delete in favor of API methods after recompile
# def get_sim(world):
#     return get_actor_by_name('DeepDriveSimulation')
# api_methods.get_sim = get_sim

def main():
    world = api_methods.get_world() or ue.get_editor_world()

    # TODO: Only reimport on file change
    # import importlib
    # importlib.reload(api_methods)

    sim = get_sim(world)
    agents = sim.GetAgentsList()
    for agent in agents:
        print('agent class name', agent.get_class().get_name())
        print('agent id', agent.GetAgentId())

        ctrlr = agent.getAgentController()
        ueprint('agent controller', ctrlr.get_name())
        try:
            ueprint('speed range start', ctrlr.m_Configuration.SpeedRange.x)
            ueprint('speed range end', ctrlr.m_Configuration.SpeedRange.y)
        except Exception as e:
            print(e)


    # ueprint('agent list', sim.GetAgentsList())

    # sim = api_methods.get_sim(world)
    # ueprint('sim object', sim)


def get_sim(world):
    sim_objs  = get_objects_of_type('DeepDriveSim_C', world)
    if len(sim_objs) > 1:
        raise ValueError('Sim is a singleton, should be only one, but got %r' % len(sim_objs))
    elif not sim_objs:
        print('Could not find DeepDriveSim object')
    return sim_objs[0]


# TODO: Move to api_methods
UE_OBJECTS_BY_TYPE = None
def get_objects_of_type(object_type, world):
    global UE_OBJECTS_BY_TYPE
    if UE_OBJECTS_BY_TYPE is None:
        UE_OBJECTS_BY_TYPE = collections.defaultdict(list)
        objects = world.all_objects()
        for o in objects:
            class_name = o.get_class().get_name()
            # ueprint('str-class', str(class_name))
            UE_OBJECTS_BY_TYPE[class_name].append(o)
    # print('\n'.join(sorted(UE_OBJECTS_BY_TYPE.keys())))
    return UE_OBJECTS_BY_TYPE[object_type]


if __name__ == '__main__':
    main()