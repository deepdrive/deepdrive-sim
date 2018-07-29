import socket
import sys
import time

import deepdrive_simulation

def cleanUp():
	pass

connected = deepdrive_simulation.connect('127.0.0.1', 9009)

if connected:
	print('Connected ...')

	try:
		mainCounter = 100000
		while mainCounter > 0:
			pass

	except KeyboardInterrupt:
		cleanUp()

	except deepdrive_simulation.connection_lost:
		print('>>>> Connection lost')

else:
	print('Connecting failed')

