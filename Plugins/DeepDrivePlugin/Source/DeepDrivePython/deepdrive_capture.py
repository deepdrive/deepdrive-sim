
import socket
import sys
import time

import numpy
import random

import deepdrive_capture
import deepdrive_client
import deepdrive_simulation

def cleanUp(clientId):
    deepdrive_client.release_agent_control(clientId)
    deepdrive_capture.close()
    deepdrive_client.close(clientId)
    deepdrive_simulation.disconnect()

connected = deepdrive_simulation.connect('127.0.0.1', 9009)

client = deepdrive_client.create('127.0.0.1', 9876)
print(client)

if client != None and 'client_id' in client:
    clientId = client['client_id']
    print('Connected ...', clientId)

    sharedMem = deepdrive_client.get_shared_memory(clientId)
    print('SharedMemName:', sharedMem[0], "Size", sharedMem[1])

    camIds = []
    camIds.append(deepdrive_client.register_camera(clientId, 60, 1024, 1024, [0.0, 0.0, 200.0], [0, 0, 0], 'MainCamera'))
    #camIds.append(deepdrive_client.register_camera(clientId, 60, 512, 256, [0.0, 0.0, 200.0], [0, 0, -60], 'FrontRight')
    #camIds.append(deepdrive_client.register_camera(clientId, 60, 512, 256, [0.0, 0.0, 200.0], [0, 0, 60], 'FrontLeft'))

    connected = deepdrive_capture.reset(sharedMem[0], sharedMem[1])
    if connected:

        try:
            while True:
                snapshot = deepdrive_capture.step()
                if snapshot:
                    if snapshot.camera_count > 0:
                        print(snapshot.cameras[0].image_data[0], snapshot.cameras[0].image_data[1], snapshot.cameras[0].image_data[2] )

                time.sleep(0.05)

            cleanUp(clientId)

        except KeyboardInterrupt:
            deepdrive_client.unregister_camera(clientId, 0)
            cleanUp(clientId)

        except deepdrive_client.connection_lost:
            print('>>>> Connection lost')
            deepdrive_capture.close()
            deepdrive_simulation.disconnect()

else:
    print('Ohh shit ...')
