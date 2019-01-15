# TODO: Do each stage - commit to stage named tags. Then allow building starting at any stage
import os
import sys
import logging as log

import docker
from Packaging import utils

log.basicConfig(format='%(asctime)-15s - %(clientip)s - %(levelname)s - %(message)s"', level=log.INFO)

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
SIM_DIR = os.path.dirname(os.path.dirname(CUR_DIR))
TAG = 'deepdriveio/deepdrive-ue4'
DOCKER_DIR = '/home/ue4/deepdrive-sim'
VOLUMES = {SIM_DIR: {'bind': '/home/ue4/deepdrive-sim', 'mode': 'rw'}}


def main():
    client = docker.from_env()

    def package():
        # Using native docker interface to stream logs
        utils.run_command_async('docker run -v {SIM_DIR}:{DOCKER_DIR} {TAG} python3 -u package.py'.format(
            SIM_DIR=SIM_DIR, DOCKER_DIR=DOCKER_DIR, TAG=TAG))
        commit()

    def clean():
        # Using native docker interface to stream logs
        utils.run_command_async('docker build -t {TAG} .'.format(TAG=TAG))

    def commit():
        latest = client.containers.list(limit=1)[0]
        latest.commit(TAG)

    if (sys.argv[1:]+[None])[0] == '--commit-only':
        commit()
    else:
        try:
            package()
        except:
            log.error('**** Error packaging in cached container - trying from scratch')
            clean()
            package()


if __name__ == '__main__':
    main()
