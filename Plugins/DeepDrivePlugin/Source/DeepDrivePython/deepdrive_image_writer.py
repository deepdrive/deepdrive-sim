
from PIL import Image

import socket
import sys
import time

import numpy

import deepdrive_capture
import deepdrive_client



def cleanUp(clientId):
	deepdrive_client.release_agent_control(clientId)
	deepdrive_capture.close()
	deepdrive_client.close(clientId)

print('ksdfjo')
client = deepdrive_client.create('127.0.0.1', 9876)
print('ksdfjo')
print(client)

if client != None and 'client_id' in client:
	clientId = client['client_id']
	print('Connected ...', clientId)

	sharedMem = deepdrive_client.get_shared_memory(clientId)
	print('SharedMemName:', sharedMem[0], "Size", sharedMem[1])

	cam0Size = (384, 384)
	deepdrive_client.register_camera(clientId, 60, cam0Size[0], cam0Size[1], [0.0, 0.0, 200.0], [0, 0, 0], 'MainCamera')

	cam0Image = Image.new("RGB", cam0Size )

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
		try:

			while True:
				snapshot = deepdrive_capture.step()
				if snapshot:
				#	print(snapshot.capture_timestamp, snapshot.sequence_number, snapshot.speed, snapshot.is_game_driving, snapshot.camera_count, len(snapshot.cameras) )
				#	for c in snapshot.cameras:
				#		print('Id', c.id, c.capture_width, 'x', c.capture_height)

					src = snapshot.cameras[0].image_data.astype(numpy.float32)
					srcInd = 0
					pixels = cam0Image.load()
					for i in range(cam0Size[0]):
						for j in range(cam0Size[1]):
							pixels[j, i] = (int(src[srcInd] * 255), int(src[srcInd + 1] * 255), int(src[srcInd + 2] * 255))
							srcInd = srcInd + 3

					cam0Image.save('cam0.png')
					print(snapshot.capture_timestamp)

				time.sleep(0.05)

			cleanUp(clientId)

		except KeyboardInterrupt:
			cleanUp(clientId)

		except deepdrive_client.connection_lost:
			print('>>>> Connection lost')

else:
	print('Ohh shit ...')
