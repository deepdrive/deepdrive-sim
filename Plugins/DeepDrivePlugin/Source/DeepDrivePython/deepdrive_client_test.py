import socket
import sys
import time

import deepdrive
import deepdrive_client

clientId = deepdrive_client.create('127.0.0.1', 9876)

def cleanUp(clientId):
	deepdrive_client.release_agent_control(clientId)
	deepdrive.close()
	deepdrive_client.close(clientId)
	clientId = 0


if clientId > 0:
	print('Connected ...', clientId)

	sharedMem = deepdrive_client.get_shared_memory(clientId)
	print('SharedMemName:', sharedMem[0], "Size", sharedMem[1])

	deepdrive_client.register_camera(clientId, 60, 1024, 1024, [1.0,2,3])

	connected = deepdrive.reset(sharedMem[0], sharedMem[1])
	if connected:
		print('Capture connected')
		print('------------------------')
		print('')
		reqCounter = 0
		try:
			mainCounter = 100000
			while mainCounter > 0:
				print('Taking over control .....')
				ctrlAcquired = deepdrive_client.request_agent_control(clientId)
				print(reqCounter, ': Control acquired', ctrlAcquired)
				counter = 3000
				while counter > 0:
					snapshot = deepdrive.step()
					if snapshot:
					#	print(snapshot.capture_timestamp, snapshot.sequence_number, snapshot.speed, snapshot.is_game_driving, snapshot.camera_count, len(snapshot.cameras) )
					#	for c in snapshot.cameras:
					#		print('Id', c.id, c.capture_width, 'x', c.capture_height)

						deepdrive_client.set_control_values(clientId, 0.0, 1.0, 0.0, 0)

					time.sleep(0.001)
					counter = counter - 1

				print('Resetting agent .....')
				deepdrive_client.reset_agent(clientId)
				print(reqCounter, ': Releasing control .....')
				deepdrive_client.release_agent_control(clientId)
				print('------------------------')
				print('')
				time.sleep(3.0)
				mainCounter = mainCounter - 1
				reqCounter = reqCounter + 1

			cleanUp(clientId)

		except KeyboardInterrupt:
			cleanUp(clientId)


if clientId > 0:
	deepdrive_client.close(clientId)

