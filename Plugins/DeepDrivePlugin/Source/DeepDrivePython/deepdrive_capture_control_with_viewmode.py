import argparse
import deepdrive_capture
import deepdrive_client
import time
import platform


def get_client():
    client = deepdrive_client.create('127.0.0.1', 9876)
    if client is not None and 'client_id' in client:
        client_id = client['client_id']
        print('Obtained client ...', client_id)
    return client


def loop(client):
    shared_mem = deepdrive_client.get_shared_memory(client['client_id'])
    print('SharedMemName:', shared_mem[0], "Size", shared_mem[1])

    snapshots = []
    deepdrive_client.register_camera(client['client_id'], 60, 1024, 1024, [0.0, 0.0, 200.0], [0, 0, 0], 'MainCamera')
    connected = deepdrive_capture.reset(shared_mem[0], shared_mem[1])
    if connected:
        print('capture connected')
        deepdrive_client.set_view_mode(client['client_id'], -1, '')
        try:
            while True:
                snapshot = deepdrive_capture.step()
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

                    capture_timestamp = snapshot.capture_timestamp
                    capture_sequence_number = snapshot.sequence_number
                    brake = 0.0
                    handbrake = 0.0

                    steering = snapshot.steering + 0.01
                    throttle = snapshot.throttle

                    deepdrive_client.set_control_values(client['client_id'], steering=steering,
                                                        throttle=throttle,
                                                        brake=brake, handbrake=0)

                time.sleep(0.001)

        except KeyboardInterrupt:
            deepdrive_capture.close()
            deepdrive_client.close()


if __name__ == '__main__':
    loop(get_client())
