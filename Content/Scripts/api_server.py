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
import api_methods


API_PORT = 5657
API_TIMEOUT_MS = 5000


class ApiServer(object):
    def __init__(self):
        self.socket = None
        self.poller = None
        self.context = None
        self.env = None
        self.conn_string = "tcp://*:%s" % API_PORT
        self.api = None

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

        # Creating a new socket on timeout is not working when other ZMQ
        # connections are present in the process.
        # socket.RCVTIMEO = API_TIMEOUT_MS
        # socket.SNDTIMEO = API_TIMEOUT_MS

        socket.bind(self.conn_string)

        print('Registering API server poller')

        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN | zmq.POLLOUT)

        self.socket = socket
        self.poller = poller

        return socket

    async def run(self):
        print('Getting API methods')
        self.api = api_methods.Api()
        print('Creating API server socket')
        await self.create_socket()
        print('Unreal API server started at %s v0.8' % self.conn_string)
        while True:
            try:
                await self.check_for_messages()
            except asyncio.CancelledError:
                print('Server shut down signal detected')
                self.close()
                break

    async def check_for_messages(self):
        try:
            msg = await self.socket.recv()
            await self.eval(msg)
        except zmq.error.Again:
            print('Waiting for client')
            await self.create_socket()

    async def wait_for_msg(self):
        while True:
            socks = dict(self.poller.poll())
            if self.socket in socks and \
                    socks[self.socket] == zmq.POLLOUT|zmq.POLLIN:
                msg = await self.socket.recv()
                return msg
            else:
                time.sleep(1e-6)

    async def eval(self, msg):
        try:
            method_name, args, kwargs = pyarrow.deserialize(msg)  # ~100 us
            fn = getattr(self.api, method_name)
            start_call = time.time()
            resp = fn(*args, **kwargs)
            print('%s took %rs' % (method_name, time.time() - start_call))
        except AttributeError:
            message = traceback.format_exc() + '\n' + \
                      'HINT: Try resarting the simulator if you\'ve ' \
                      'just added a new method'
            await self.socket.send(
                serialize({'success': False, 'result': message}))
        except Exception:
            await self.socket.send(
                serialize({'success': False, 'result': traceback.format_exc()}))
        else:
            start_send = time.time()
            await self.socket.send(
                serialize({'success': True, 'result': resp}))  # ~ 0.5ms
            print('send took %rs' % (time.time() - start_send))

    def __del__(self):
        self.close()

    def close(self):
        print('Closing lambda server')
        try:
            self.poller.unregister(self.socket)
            self.socket.close()
        except Exception as e:
            print('Error closing lambda server zmq socket' + str(e))
        finally:
            try:
                self.context.destroy()
            except Exception as e:
                print('Error destroying zmq context in lambda server')


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
    # Just for testing - dummy_actor starts server in-game to ensure world
    # is loaded prior
    print('Testing event loop server')
    loop = asyncio.get_event_loop()
    loop.set_debug(enabled=True)
    loop.run_until_complete(ApiServer().run())


if __name__ == '__main__':
    print([getattr(api_methods, a) for a in dir(api_methods)
           if isinstance(getattr(api_methods, a), types.FunctionType)])

    start_server_test()
