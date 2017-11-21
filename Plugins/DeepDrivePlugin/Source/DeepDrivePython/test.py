import argparse
import deepdrive
import time
import platform

if platform.system() == 'Linux':
    connected = deepdrive.reset('/tmp/deepdrive_shared_memory', 157286400)
elif platform.system() == 'Windows':
    connected = deepdrive.reset('Local\DeepDriveCapture', 157286400)


def test_loop():
    parser = argparse.ArgumentParser(description='Example of how to access Unreal data')
    parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    parser.add_argument('-s', '--show', action='store_true', help='Show images - useful for debugging')
    args = parser.parse_args()
    print(connected)
    if connected:
        try:
            while True:
                snapshot = deepdrive.step()
                if snapshot:
                    print(snapshot.capture_timestamp, snapshot.sequence_number, snapshot.speed, snapshot.is_game_driving, snapshot.camera_count)
                    print(snapshot.position)
                    print(snapshot.rotation)
                    print(snapshot.velocity)
                    print(snapshot.acceleration)

                    for cc in snapshot.cameras:
                        print('  Camera:', cc.type, cc.id, cc.capture_width, 'x', cc.capture_height)
                        print('    image size', len(cc.image_data), cc.image_data[0], cc.image_data[1], cc.image_data[2])
                        print('    depth size', len(cc.depth_data))
                time.sleep(0.001)

        except KeyboardInterrupt:
            deepdrive.close()

if __name__ == '__main__':
    test_loop()
