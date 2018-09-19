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

    # this is called at every 'tick'
    def tick(self, delta_time):
        pass
        # # get current location
        # location = self.uobject.get_actor_location()
        # # increase Z honouring delta_time
        # location.z += 100 * delta_time
        # # set new location
        # self.uobject.set_actor_location(location)