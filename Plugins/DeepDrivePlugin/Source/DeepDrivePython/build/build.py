from __future__ import print_function

import argparse
import glob
import os
from subprocess import Popen, PIPE
from pathlib import Path

import config
import sys

DIR = os.path.dirname(os.path.realpath(__file__))


def run_command(cmd, cwd=None, env=None, log_filename=None, err_filename=None, err_token=None):
    print('running %s' % cmd)
    if not isinstance(cmd, list):
        cmd = cmd.split()
    cmd = filter(None, cmd)  # filter out empty items
    p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=cwd, env=env)
    result, err = p.communicate()
    if not isinstance(result, str):
        result = ''.join(map(chr, result))
    result = result.strip()
    if not isinstance(err, str):
        err = ''.join(map(chr, err))
    err = err.strip()

    if log_filename is not None:
        with open(log_filename, 'w') as log_file:
            log_file.write(result)
            print('wrote build log output to ' + os.path.abspath(log_filename))
    if err_filename is not None and err:
        with open(err_filename, 'w') as error_log_file:
            error_log_file.write(err)
            print('wrote build error output to ' + os.path.abspath(err_filename))
    if err_token is not None:
        log_lines = result.split('\n')
        err_lines = err.split('\n')
        error_strings = [s for s in log_lines + err_lines if err_token in s.lower()]
        if error_strings:
            print('Errors:\n' + '\n'.join(error_strings))
    print(result)

    if p.returncode != 0:
        raise RuntimeError(' '.join(cmd) + ' finished with error ' + err)

    return result, err


def get_egg_file(module_name):
    def f(packages):
        return glob.glob(
            os.path.join(os.path.dirname(os.path.dirname(sys.executable)),
                         'lib', 'python*', packages, module_name + '.egg-link'))

    return f('site-packages') or f('dist-packages')


def main(build_type):
    unreal_root = str(Path(DIR).parent.parent.parent.parent.parent)
    print('Unreal root %s' % unreal_root)
    env = os.environ.copy()
    py = sys.executable
    get_package_version_path = os.path.join(unreal_root, 'Packaging', 'get_package_version.py')
    env['DEEPDRIVE_VERSION'], _ = run_command('%s %s' % (py, get_package_version_path))
    print('DEEPDRIVE_VERSION is %s' % env['DEEPDRIVE_VERSION'])
    ext_root = os.path.dirname(DIR)
    print('PYPY_USERNAME is %s' % env.get('PYPI_USERNAME'))
    if build_type == 'dev':
        egg_file = get_egg_file('deepdrive')
        if egg_file:
            print('Removing ', egg_file)
            os.remove(egg_file[0])

        try:
            run_command('%s -m pip uninstall --yes '    % py + config.PACKAGE_NAME, env=env, cwd=ext_root)
        except Exception as e:
            print('Best effort uninstall method 1 of external deepdrive package failed, error was: %s' % str(e))
        try:
            run_command('pip uninstall --yes ' + config.PACKAGE_NAME, env=env, cwd=ext_root)
        except Exception as e:
            print('Best effort uninstall method 2 of external deepdrive package failed, error was: %s' % str(e))
        try:
            build_dev(env, ext_root, py)
        except Exception as e:
            msg = 'Error building, you may need to kill running\nPython processes which have imported the deepdrive' \
                  '\nmodule.'
            print("""
            
*****************************************************
*****************************************************

""" + msg + """

*****************************************************
*****************************************************

""")
            raise Exception(msg, e)

    else:
        git_dir = os.path.join(unreal_root, '.git')
        if not os.path.exists(git_dir):
            raise RuntimeError('Error: Git is required to build a distribution release. Use --type '
                               'dev to build locally.')
        env['DEEPDRIVE_BRANCH'] = (env.get('TRAVIS_BRANCH') or env.get('APPVEYOR_REPO_BRANCH') or
                                   run_command(['git', '-C', unreal_root, 'rev-parse', '--abbrev-ref', 'HEAD'])[0])
        build_pypi(build_type, env, ext_root, py, unreal_root)


def build_pypi(build_type, env, ext_root, py, sim_root):
    if env['DEEPDRIVE_BRANCH'] != 'release':
        env['DEEPDRIVE_VERSION_SEGMENT'] = '.dev0'
    else:
        env['DEEPDRIVE_VERSION_SEGMENT'] = ''

    if build_type == 'win_bdist':
        run_command('%s -u setup.py bdist_wheel' % py, env=env, cwd=ext_root)
        scripts_dir = os.path.join(env['PYTHON'], 'Scripts')
        print('DEBUG scripts dir %s' % list(os.listdir(scripts_dir)))
        for name in os.listdir(os.path.join(ext_root, 'dist')):
            if env['DEEPDRIVE_VERSION'] in name and name.endswith(".whl"):
                twine = os.path.join(scripts_dir, 'twine')
                run_command([
                    twine, 'upload',
                    os.path.join(ext_root, 'dist', name),
                    '-u', env['PYPI_USERNAME'],
                    '-p', env['PYPI_PASSWORD'],
                    '--skip-existing',
                ], env=env, cwd=ext_root)
    elif build_type == 'linux_bdist':
        def ensure_no_binaries_of_type(file_extension):
            ret = glob.glob(DIR + '/**/*' + file_extension, recursive=True)
            if ret:
                print('Error, found existing binaries, '
                      'please delete the following to package a manylinux distrubution: \n\t%s' % '\n\t'.join(ret))
            return ret

        so_files = ensure_no_binaries_of_type('.so')
        o_files = ensure_no_binaries_of_type('.o')
        if so_files or o_files:
            exit(1)

        env['PRE_CMD'] = env.get('PRE_CMD') or ''
        env['DOCKER_IMAGE'] = env.get('DOCKER_IMAGE') or 'quay.io/pypa/manylinux1_x86_64'

        # Build in CentOS to get a portable binary

        docker_options = [
            '--rm',
            '--net', 'host',
            '-e', '"DEEPDRIVE_SRC_DIR=/io"',
            '-e', 'PYPI_USERNAME',
            '-e', 'PYPI_PASSWORD',
            '-e', 'DEEPDRIVE_BRANCH',
            '-e', 'DEEPDRIVE_VERSION',
            '-e', 'DEEPDRIVE_VERSION_SEGMENT',
            '-v', sim_root + '/Plugins/DeepDrivePlugin/Source:/io', ]

        docker_positional_args = [
            env['DOCKER_IMAGE'],
            env['PRE_CMD'],
            '/io/DeepDrivePython/build/build-linux-wheels.sh']

        docker_command = ['docker', 'run'] + docker_options + \
                         docker_positional_args

        run_command(docker_command, env=env, cwd=os.path.dirname(DIR))


def build_dev(env, ext_root, py):
    run_command(
        '%s -u -m pip install -e . --upgrade --force-reinstall --ignore-installed --no-deps' % py,
        env=env, cwd=ext_root, log_filename='dev_build_log.txt', err_filename='dev_build_err.txt',
        err_token='error:')


def get_centos_py_versions():
    """
    Executed within the centos manylinux container to dynamically add
    new python versions as they become available in the container
    :return (str): String that is a space separated list version bins, i.e.
      /opt/python/cp35-cp35m/bin /opt/python/cp36-cp36m/bin
    """
    dirs = glob.glob('/opt/python/cp3*-cp3*m/bin')
    if not dir:
        raise RuntimeError("No python versions found")
    return ' '.join(['%s' % d for d in dirs])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--type', nargs='?', default='dev',
                        help='Type of build', choices=['dev', 'win_bdist',
                                                       'linux_bdist'])
    args = parser.parse_args()
    main(args.type)
