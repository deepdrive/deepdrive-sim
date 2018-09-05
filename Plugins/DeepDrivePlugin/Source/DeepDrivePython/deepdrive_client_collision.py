import socket
import sys
import time

import deepdrive_capture
import deepdrive_client

def cleanUp(clientId):
	deepdrive_client.release_agent_control(clientId)
	deepdrive_capture.close()
	deepdrive_client.close(clientId)

client = deepdrive_client.create('127.0.0.1', 9876)
print(client)

if client != None and 'client_id' in client:
	clientId = client['client_id']
	print('Connected ...', clientId)

	sharedMem = deepdrive_client.get_shared_memory(clientId)
	print('SharedMemName:', sharedMem[0], "Size", sharedMem[1])

	deepdrive_client.register_camera(clientId, 60, 1024, 1024, [0.0, 0.0, 200.0], [0, 0, 0], 'MainCamera')

	# deepdrive_client.register_camera(clientId, 60, 512, 256, [0.0, 0.0, 200.0], [0.0, 0.0, 60.0], 'FrontRight')
	# deepdrive_client.register_camera(clientId, 60, 512, 256, [0.0, 0.0, 200.0], [0.0, 0.0, -60.0], 'FrontLeft')
	#deepdrive_client.register_camera(clientId, 60, 512, 256, [0.0, 0.0, 200.0], [0.0, 0.0, 120.0])
	#deepdrive_client.register_camera(clientId, 60, 512, 256, [0.0, 0.0, 200.0], [0.0, 0.0, -120.0])
	#deepdrive_client.register_camera(clientId, 60, 512, 256, [0.0, 0.0, 200.0], [0.0, 0.0, 180.0])
	#deepdrive_client.register_camera(clientId, 60, 512, 256, [0.0, 0.0, 200.0], [0.0, 0.0, -60.0])
	#deepdrive_client.register_camera(clientId, 60, 512, 256, [0.0, 0.0, 200.0], [0.0, 0.0, -60.0])

	connected = deepdrive_capture.reset(sharedMem[0], sharedMem[1])
	if connected:
		print('Capture connected')
		print('------------------------')
		print('')
		reqCounter = 0
		try:
			while True:
				snapshot = deepdrive_capture.step()
				if snapshot:
					print(snapshot.last_collision.time_utc, snapshot.last_collision.collision_location, snapshot.last_collision.collidee_velocity, snapshot.last_collision.collision_normal)

				time.sleep(0.1)

			cleanUp(clientId)

		except KeyboardInterrupt:
			cleanUp(clientId)

		except deepdrive_client.connection_lost:
			print('>>>> Connection lost')

else:
	print('Ohh shit ...')

