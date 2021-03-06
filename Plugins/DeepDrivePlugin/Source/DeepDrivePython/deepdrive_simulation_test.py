import socket
import sys
import time

import deepdrive_simulation

def cleanUp():
	deepdrive_simulation.disconnect()

connected = deepdrive_simulation.connect('127.0.0.1', 9009)

if connected:
	print('Connected ...')

	gfxSettings = deepdrive_simulation.SimulationGraphicsSettings();
	gfxSettings.shadow_quality = 0
	hour = 9

	try:
		mainCounter = 100000
		while mainCounter > 0:
			deepdrive_simulation.set_date_and_time(hour=hour)
			deepdrive_simulation.set_graphics_settings(gfxSettings)
			#hour = hour + 1
			if hour > 23:
				hour = 0
			gfxSettings.shadow_quality = gfxSettings.shadow_quality + 1
			if gfxSettings.shadow_quality > 5:
				gfxSettings.shadow_quality = 0;
			time.sleep(3)

	except KeyboardInterrupt:
		cleanUp()

	except deepdrive_simulation.connection_lost:
		print('>>>> Connection lost')

else:
	print('Connecting failed')

