import unreal_engine as ue

"""Liason class between Python and Unreal that allows hooking into the world tick"""

ue.log('Loading DummyPyActor')


class DummyPyActor:

    def __init__(self):
        self.worlds = None

    # this is called on game start
    def begin_play(self):
        ue.log('Begin Play on DummyPyActor class')
        self.worlds = ue.all_worlds()
        ue.log('All worlds length ' + str(len(self.worlds)))


        # controllers = [(a.get_full_name(), a)
        #                for a in ue.all_worlds()[0].all_actors() if 'controller' in a.get_full_name().lower()]
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
        pass
        # py.cmd print([a.get_full_name() for a in ue.all_worlds()[0].all_actors()])
        #  # get current location
        # location = self.uobject.get_actor_location()
        # # increase Z honouring delta_time
        # location.z += 100 * delta_time
        # # set new location
        # self.uobject.set_actor_location(location)

