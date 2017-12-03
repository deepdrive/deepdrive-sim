from __future__ import print_function

import argparse
from subprocess import Popen, PIPE
import os
import sys
DIR = os.path.dirname(os.path.realpath(__file__))


def run_command(cmd, cwd=None, env=None, stream=False):
    p = Popen(cmd.split(' '), stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=cwd, env=env)
    if stream:
        for c in iter(lambda: p.stdout.read(1), b''):
            sys.stdout.write(''.join(map(chr, c)))
    else:
        result, err = p.communicate()
        result = ''.join(map(chr, result)).strip()
        if p.returncode != 0:
            raise RuntimeError(cmd + ' finished with error ' + ''.join(map(chr, err)).strip())
        return result

def main(build_type):
    root = run_command('git rev-parse --show-toplevel')
    env = os.environ.copy()
    get_package_version_path = os.path.join(root, 'Packaging', 'get_package_version.py')
    env['DEEPDRIVE_VERSION'] = run_command('python ' + get_package_version_path)
    env['DEEPDRIVE_BRANCH'] = run_command('git rev-parse --abbrev-ref HEAD')
    if build_type == 'dev':
        run_command('python -u setup.py install', env=env, cwd=os.path.dirname(DIR), stream=True)
    elif build_type == 'win_dist':
        run_command('python -u setup.py bdist_wheel', env=env, cwd=os.path.dirname(DIR), stream=True)

    # Use build.sh for Unix

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--type', nargs='?', default='dev', help='Type of build [dev, win_dist]')
    args = parser.parse_args()
    main(args.type)
