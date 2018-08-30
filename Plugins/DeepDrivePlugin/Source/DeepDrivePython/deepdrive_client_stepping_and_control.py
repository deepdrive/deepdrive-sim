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

	deepdrive_client.register_camera(clientId, 60, 512, 256, [0.0, 0.0, 200.0], [0.0, 0.0, 60.0], 'FrontRight')
	deepdrive_client.register_camera(clientId, 60, 512, 256, [0.0, 0.0, 200.0], [0.0, 0.0, -60.0], 'FrontLeft')
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
			print('Activating synchronous stepping .....')
			syncStepping = deepdrive_client.activate_synchronous_stepping(clientId)
			print('Done', syncStepping)

			mainCounter = 100000
			while mainCounter > 0:
				print('Taking over control .....')
				ctrlAcquired = deepdrive_client.request_agent_control(clientId)
				print(reqCounter, ': Control acquired', ctrlAcquired)
				counter = 25
				while counter > 0:
					snapshot = deepdrive_capture.step()
					if snapshot:
					#	print(snapshot.capture_timestamp, snapshot.sequence_number, snapshot.speed, snapshot.is_game_driving, snapshot.camera_count, len(snapshot.cameras) )
					#	for c in snapshot.cameras:
					#		print('Id', c.id, c.capture_width, 'x', c.capture_height)
						#print(snapshot.last_collision.time_utc, snapshot.last_collision.collidee_velocity)
						if snapshot.camera_count > 0:
							print(snapshot.cameras[0].image_data[0], snapshot.cameras[0].image_data[1], snapshot.cameras[0].image_data[2] )

					seqNr = deepdrive_client.advance_synchronous_stepping(clientId, 0.125, 0.2, 1.0, 0.0, 0)
					print('Advanced', counter, seqNr)
					time.sleep(0.05)
					counter -= 1

				# print('Resetting agent .....')
				# deepdrive_client.reset_agent(clientId)
				
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

		except deepdrive_client.connection_lost:
			print('>>>> Connection lost')

else:
	print('Ohh shit ...')

