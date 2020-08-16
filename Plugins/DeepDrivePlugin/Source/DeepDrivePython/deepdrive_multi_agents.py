import socket
import sys
import time

import deepdrive_simulation

def cleanUp():
	print('Releasing control .....')
	deepdrive_simulation.multi_agent.release_control([1, 2, 3])

	deepdrive_simulation.disconnect()

connected = deepdrive_simulation.connect('127.0.0.1', 9009)

if connected:
	print('Connected to DeepDriveSimulation ...')

	agentsList = deepdrive_simulation.multi_agent.get_agents_list()
	print(agentsList)

	print('Taking over control .....')
	deepdrive_simulation.multi_agent.request_control([1, 2, 3])

	try:
		while True:
			deepdrive_simulation.multi_agent.set_control_values( [{'id':1, 'throttle':0.3, 'steering': 0.5}, {'id':2, 'throttle':0.5, 'steering':-0.2}, {'id':3, 'throttle':0.5, 'steering':0.2}] )
			time.sleep(0.05)

	except KeyboardInterrupt:
		cleanUp()

	except deepdrive_simulation.connection_lost:
		print('>>>> Connection lost')
