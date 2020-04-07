import socket
import sys
import time

import deepdrive_simulation

def cleanUp():
	deepdrive_simulation.disconnect()

connected = deepdrive_simulation.connect('127.0.0.1', 9009)

if connected:
	print('Connected ...')

	deepdrive_simulation.set_sun_simulation_speed(1000)

	try:
		while True:
			time.sleep(0.1)

	except KeyboardInterrupt:
		cleanUp()

	except deepdrive_simulation.connection_lost:
		print('>>>> Connection lost')

else:
	print('Connecting failed')

