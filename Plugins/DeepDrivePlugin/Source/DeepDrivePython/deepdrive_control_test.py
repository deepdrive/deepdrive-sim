import argparse
import deepdrive_control
import time
import platform

if platform.system() == 'Linux':
    created = deepdrive_control.reset('/tmp/deepdrive_control', 1048576)
elif platform.system() == 'Windows':
    created = deepdrive_control.reset('Local\DeepDriveControl', 1048576)


def loop():
    # parser = argparse.ArgumentParser(description='Example of how to control Unreal')
    # parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    # args = parser.parse_args()
    print(created)
    if created:
        try:
            ctrl = deepdrive_control.DeepDriveControl()
            ctrl.steering = 0.0
            ctrl.throttle = 1.0
            ctrl.brake = 0.0
            ctrl.handbrake = 0.0
            ctrl.is_game_driving = False
            while True:

                time.sleep(0.5)

                ctrl.steering = ctrl.steering + 0.01

                deepdrive_control.send_control(ctrl)


        except KeyboardInterrupt:
            deepdrive_control.close()

if __name__ == '__main__':
    loop()
