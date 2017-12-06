from __future__ import print_function

import argparse
from subprocess import Popen, PIPE
import os
import sys
DIR = os.path.dirname(os.path.realpath(__file__))


def run_command(cmd, cwd=None, env=None, stream=False):
    print('running %s' % cmd)
    p = Popen(cmd.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=cwd, env=env)
    if stream:
        for c in iter(lambda: p.stdout.read(1), b''):
            sys.stdout.write(''.join(map(chr, c)))
    else:
        result, err = p.communicate()
        if isinstance(result, bytes):
            result = ''.join(map(chr, result)).strip()
        if p.returncode != 0:
            if isinstance(err, bytes):
                err = ''.join(map(chr, err)).strip()
            raise RuntimeError(cmd + ' finished with error ' + ''.join(map(chr, err)).strip())
        return result

def main(build_type):
    unreal_root = run_command('git rev-parse --show-toplevel')
    env = os.environ.copy()
    get_package_version_path = os.path.join(unreal_root, 'Packaging', 'get_package_version.py')
    env['DEEPDRIVE_VERSION'] = run_command('python ' + get_package_version_path)
    env['DEEPDRIVE_BRANCH'] = (env.get('TRAVIS_BRANCH') or env.get('APPVEYOR_REPO_BRANCH') or
                               run_command('git rev-parse --abbrev-ref HEAD'))
    ext_root = os.path.dirname(DIR)
    if build_type == 'dev':
        run_command('python -u setup.py install', env=env, cwd=ext_root, stream=True)
    elif build_type == 'win_bdist':
        run_command('python -u setup.py bdist_wheel', env=env, cwd=ext_root, stream=True)
        if env['DEEPDRIVE_BRANCH'] == 'release':
            for name in os.listdir(os.path.join(ext_root, 'dist')):
                if env['DEEPDRIVE_VERSION'] in name and name.endswith(".whl"):
                    run_command('twine upload ' + os.path.join(ext_root, 'dist', name), env=env, cwd=ext_root,
                                stream=True)
    elif build_type == 'linux_bdist':
        env['PRE_CMD'] = env.get('PRE_CMD') or ''
        env['DOCKER_IMAGE'] = env.get('DOCKER_IMAGE') or 'quay.io/pypa/manylinux1_x86_64'

        # Build in CentOS to get a portable binary
        run_command('docker run --rm -e "DEEPDRIVE_SRC_DIR=/io" \
            -e PYPI_USERNAME \
            -e PYPI_PASSWORD \
            -e DEEPDRIVE_BRANCH \
            -e DEEPDRIVE_VERSION \
            -v ${ROOT}/Plugins/DeepDrivePlugin/Source:/io \
            ${DOCKER_IMAGE} \
            ${PRE_CMD} \
            /io/DeepDrivePython/build/build-linux-wheels.sh  ', env=env, cwd=os.path.dirname(DIR), stream=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--type', nargs='?', default='dev', help='Type of build', choices=['dev', 'win_bdist',
                                                                                           'linux_bdist'])
    args = parser.parse_args()
    main(args.type)
