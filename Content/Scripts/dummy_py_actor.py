import time

import unreal_engine as ue

"""Liason class between Python and Unreal that allows hooking into the world tick"""

ue.log('Loading DummyPyActor')


class DummyPyActor:

    def __init__(self):
        self.worlds = None
        self.world = None
        self.ai_controllers = None

    def _find_world(self):
        self.worlds = ue.all_worlds()
        # ue.log('All worlds length ' + str(len(self.worlds)))

        # print([w.get_full_name() for w in self.worlds])

        if hasattr(ue, 'get_editor_world'):
            # ue.log('Detected Unreal Editor')
            sim_world = ue.get_editor_world()
        else:
            # ue.log('Determined we are in a packaged game')
            sim_world = self.uobject.get_current_level()
        # print(sim_world)
        # print(sim_world.get_full_name())

        sim_demo_worlds = [w for w in self.worlds if 'DeepDriveSim_Demo.DeepDriveSim_Demo' in w.get_full_name()]

        # print('sim demo worlds are ')
        # print(sim_demo_worlds)

        possible_worlds = [sim_world] + sim_demo_worlds

        for w in possible_worlds:
            self.ai_controllers = [a for a in w.all_actors()
                                   if 'localaicontroller_' in a.get_full_name().lower()]
            if self.ai_controllers:
                self.world = w
                print('FOUND WORLD WITH CONTROLLER !!!!!!!!!!!!!!!!!')
                break
        else:
            # print('NO WORLDS WITH CONTROLLER')
            pass

        return self.world

    # this is called on game start
    def begin_play(self):
        ue.log('Begin Play on DummyPyActor class')
        self._find_world()

        # controllers = [(a.get_full_name(), a)
        #                for a in sim_world.all_actors() if 'localaicontroller_' in a.get_full_name().lower()]
        # print(controllers)

        # try:
        #     print(self.get_current_level())
        # except Exception as e:
        #     print(e)

        # controllers = [(a.get_full_name(), a)
        #                for a in sim_world.all_actors() if 'localaicontroller_' in a.get_full_name().lower()]
        #
        # print(controllers)
        # controller = controllers[-1][1]
        # print(dir(controller))
        # print(controller.functions())
        # print(controller.getIsPassing())

        #
        # print(controllers)
        #
        # local_ai_controller = controllers[1][1]
        #
        # from unreal_engine.classes import DeepDriveAgentLocalAIController
        #
        # # print(dir(ue.all_worlds()[0]))
        #
        # print(local_ai_controller.get_class())
        # print(local_ai_controller.__class__)
        # print(local_ai_controller.functions())
        # print(local_ai_controller.GetActorBounds())
        #
        # # print(ue.all_worlds()[0].get_components_by_class(DeepDriveAgentLocalAIController))
        #
        # agents = [(a.get_full_name(), a)
        #           for a in ue.all_worlds()[0].all_actors() if 'agent' in a.get_full_name().lower()]
        #
        # print(agents)
        #
        # print(agents[0][1].ChaseCamera)

    # this is called at every 'tick'
    def tick(self, delta_time):
        if self.world is None:
            self._find_world()

        for controller in self.ai_controllers:
            if controller.getIsPassing():
                print('IS PASSING!!!!!!!!!!!!!!!!!!!!!!!')

        # py.cmd print([a.get_full_name() for a in ue.all_worlds()[0].all_actors()])
        #  # get current location
        # location = self.uobject.get_actor_location()
        # # increase Z honouring delta_time
        # location.z += 100 * delta_time
        # # set new location
        # self.uobject.set_actor_location(location)

