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
	deepdrive_simulation.set_sun_simulation_speed(1000)
	hour = 5
	speed = 10000

	try:
		mainCounter = 100000
		while mainCounter > 0:
			#deepdrive_simulation.set_date_and_time(hour=hour)
			deepdrive_simulation.set_sun_simulation_speed(speed)
			deepdrive_simulation.reset_simulation()
			if speed > 0:
				speed = 0
			else:
				speed = 10000
			hour = hour + 1
			if hour > 23:
				hour = 0
			time.sleep(1)

	except KeyboardInterrupt:
		cleanUp()

	except deepdrive_simulation.connection_lost:
		print('>>>> Connection lost')

else:
	print('Connecting failed')

