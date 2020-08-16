import socket
import sys
import time

import deepdrive_simulation

def cleanUp():
	deepdrive_simulation.disconnect()

connected = deepdrive_simulation.connect('127.0.0.1', 9009)

if connected:
	print('Connected ...')

	time.sleep(1)
	hour = 5
	minute = 0

	try:
		mainCounter = 100000
		while mainCounter > 0:
			deepdrive_simulation.set_date_and_time(hour=hour, minute=minute)
			minute = minute + 1
			if minute > 59:
				minute = 0
				hour = hour + 1
				if hour > 23:
					hour = 0
			time.sleep(0.1)

	except KeyboardInterrupt:
		cleanUp()

	except deepdrive_simulation.connection_lost:
		print('>>>> Connection lost')

else:
	print('Connecting failed')

