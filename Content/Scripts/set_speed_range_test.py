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

def main():
    world = api_methods.get_world() or ue.get_editor_world()

    # TODO: Only reimport on file change
    # import importlib
    # importlib.reload(api_methods)

    sim = api_methods.get_sim(world)
    agents = sim.GetAgentsList()
    for agent in agents:
        print('agent class name', agent.get_class().get_name())
        print('agent id', agent.GetAgentId())

        ctrlr = agent.getAgentController()
        ueprint('agent controller', ctrlr.get_name())
        try:
            config = ctrlr.m_Configuration
            ueprint('speed range start1', config.SpeedRange.x)
            ueprint('speed range end1', config.SpeedRange.y)
            ctrlr.m_Configuration.SpeedRange.x = 70
            ctrlr.m_Configuration.SpeedRange.y = 90
            ueprint('after speed range start', config.SpeedRange.x)
            ueprint('after speed range end', config.SpeedRange.y)

        except Exception as e:
            print(e)


    # ueprint('agent list', sim.GetAgentsList())

    # sim = api_methods.get_sim(world)
    # ueprint('sim object', sim)




if __name__ == '__main__':
    main()