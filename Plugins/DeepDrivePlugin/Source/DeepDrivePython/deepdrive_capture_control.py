import argparse
import deepdrive
import deepdrive_control
import time
import platform


def connect():
    if platform.system() == 'Linux':
        connected = deepdrive.reset('/tmp/deepdrive_shared_memory', 157286400)
    elif platform.system() == 'Windows':
        connected = deepdrive.reset('Local\DeepDriveCapture', 157286400)
    return connected


def loop():
    # parser = argparse.ArgumentParser(description='Example of how to control Unreal')
    # parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    # args = parser.parse_args()

    connected = False
    while not connected:
        connected = connect()
        if connected:
            print('Capture connected')
        else:
            time.sleep(0.25)

    snapshots = []
    try:
        while True:
            snapshot = deepdrive.step()
            if snapshot:
                snapshots.append(snapshot)
                for s in snapshots:
                    for c in s.cameras:
                        print(c.image_data)
                print(snapshot.capture_timestamp, snapshot.sequence_number, snapshot.speed, snapshot.is_game_driving, snapshot.camera_count)
                print(snapshot.position)
                print(snapshot.rotation)
                print(snapshot.velocity)
                print(snapshot.acceleration)

                ctrl = deepdrive_control.DeepDriveControl()
                ctrl.capture_timestamp = snapshot.capture_timestamp
                ctrl.capture_sequence_number = snapshot.sequence_number
                ctrl.brake = 0.0
                ctrl.handbrake = 0.0
                ctrl.is_game_driving = False

                ctrl.steering = snapshot.steering + 0.01
                ctrl.throttle = snapshot.throttle

                deepdrive_control.send_control(ctrl)

            time.sleep(0.001)

    except KeyboardInterrupt:
        deepdrive.close()
        deepdrive_control.close()

if __name__ == '__main__':
    if platform.system() == 'Linux':
        created = deepdrive_control.reset('/tmp/deepdrive_control', 1048576)
    elif platform.system() == 'Windows':
        created = deepdrive_control.reset('Local\DeepDriveControl_1', 1048576)

    if created:
        loop()
