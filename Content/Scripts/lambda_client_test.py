from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import zmq
import pyarrow

API_PORT = 5657


class LambdaClient(object):
    def __init__(self, **kwargs):
        self.socket = None
        self.last_obz = None
        self.create_socket()
        # self._send(m.START, kwargs=kwargs)

    def _send(self, expression, local_vars=None):
        local_vars = local_vars or {}
        try:
            msg = pyarrow.serialize([expression, local_vars]).to_buffer()
            self.socket.send(msg)
            return pyarrow.deserialize(self.socket.recv())
        except zmq.error.Again:
            print('Waiting for server')
            self.create_socket()
            return None

    def create_socket(self):
        if self.socket:
            self.socket.close()
        context = zmq.Context()
        socket = context.socket(zmq.PAIR)

        # Creating a new socket on timeout is not working when other ZMQ connections are present in the process.
        # socket.RCVTIMEO = c.API_TIMEOUT_MS
        # socket.SNDTIMEO = c.API_TIMEOUT_MS

        socket.connect("tcp://localhost:%s" % API_PORT)
        self.socket = socket
        return socket

    def close(self):
        self.socket.close()


def main():
    client = LambdaClient()
    answer = client._send('x**2', {'x': 2})


if __name__ == '__main__':
    main()
