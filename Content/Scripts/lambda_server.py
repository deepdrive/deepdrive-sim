import asyncio
import time
import traceback
import re
import types

try:
    import unreal_engine as ue
except ImportError:
    print('Cannot import unreal engine')

import pyarrow
import zmq.asyncio
import zmq
import api_methods as api


API_PORT = 5657
API_TIMEOUT_MS = 5000


class LambdaServer(object):
    def __init__(self):
        self.socket = None
        self.context = None
        self.env = None
        self.conn_string = "tcp://*:%s" % API_PORT

    async def create_socket(self):
        if self.socket is not None:
            print('Closed server socket')
            self.socket.close()
        if self.context is not None:
            print('Destroyed context')
            self.context.destroy()

        self.context = zmq.asyncio.Context()

        # noinspection PyUnresolvedReferences
        socket = self.context.socket(zmq.PAIR)

        # Creating a new socket on timeout is not working when other ZMQ connections are present in the process.
        # socket.RCVTIMEO = API_TIMEOUT_MS
        # socket.SNDTIMEO = API_TIMEOUT_MS

        socket.bind(self.conn_string)
        self.socket = socket
        return socket

    async def run(self, world):
        await self.create_socket()
        print('Unreal Lambda server started at %s v0.5' % self.conn_string)
        while True:
            try:
                await self.check_for_messages(world)
            except asyncio.CancelledError:
                print('Server shut down signal detected')
                self.close()
                break

    async def check_for_messages(self, world):
        try:
            _locals = {'world': world}
            add_api_methods(_locals)
            msg = await self.socket.recv()
            expression_str, local_vars = pyarrow.deserialize(msg)
            _locals.update(local_vars)
            await self.eval(expression_str, _locals)

            """
            TODO: support simpler RPC as well
            ```
                method_name, args, kwargs = pyarrow.deserialize(msg)
                fn = eval(method_name)
                resp = fn(*args, **kwargs)
                self.socket.send(pyarrow.serialize(resp).to_buffer())
            ```                
            """

        except zmq.error.Again:
            print('Waiting for client')
            await self.create_socket()



    async def eval(self, expression_str, local_vars):
        try:
            # Expression can be something like [(a.get_full_name(), a) for a in world.all_actors() if 'localaicontroller_' in a.get_full_name().lower()]
            resp = eval(expression_str, None, local_vars)
        except Exception:
            await self.socket.send(serialize({'success': False, 'result': traceback.format_exc()}))
        else:
            await self.socket.send(serialize({'success': True, 'result': resp}))

    def __del__(self):
        self.close()

    def close(self):
        print('Closing lambda server')
        try:
            self.socket.close()
        except Exception as e:
            print('Error closing lambda server ' + str(e))


def event_loop_thread(world, loop):
    # loop.set_debug(enabled=True)
    loop.run_until_complete(LambdaServer().run(world=world))


def add_api_methods(_locals):
    for attr_name in dir(api):
        attr = getattr(api, attr_name)
        if isinstance(attr, types.FunctionType):
            _locals[attr_name] = attr


def serialize(obj):
    try:
        ret = pyarrow.serialize(obj).to_buffer()
    except pyarrow.lib.SerializationCallbackError:
        print('Could not serialize with pyarrow - falling back to str(obj)')
        ret = pyarrow.serialize({'success': True, 'result': str(obj)}).to_buffer()
    except:
        ret = {'success': False, 'result': traceback.format_exc()}
    return ret


def start_server_test():
    # Just for testing - dummy_actor starts server in-game to ensure world is loaded prior
    print('Testing event loop server')
    loop = asyncio.get_event_loop()
    loop.set_debug(enabled=True)
    loop.run_until_complete(LambdaServer().run(world=None))


if __name__ == '__main__':
    print([getattr(api, a) for a in dir(api)
      if isinstance(getattr(api, a), types.FunctionType)])

