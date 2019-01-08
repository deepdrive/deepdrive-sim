# Goal: Continuous integration packaging in docker, upload to s3, build env image, test benchmark in image

# Usage: See Dockerfile

# Dev

# If you're debugging python and want to commit filesystem changes before some exception you can do this
# to commit the latest container
# docker commit `docker ps --latest --format "{{.ID}}"` deepdriveio/deepdrive-ue4
import functools
import os
import shutil
import time
from glob import glob
import logging as log

import pypip

PYTHON_REQUIREMENTS = [
    {'module': 'git', 'pip': 'gitpython'},
    {'module': 'github', 'pip': 'PyGithub'},
    'clint',
    'requests',
    'six',
    'docker',
    'sarge'
]

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
HOME_DIR = os.environ['HOME']
SIM_DIR = os.path.join(HOME_DIR, 'deepdrive-sim')
SIM_GITHUB = 'https://github.com/deepdrive/deepdrive-sim'
SIM_BRANCH = 'deepdrive_sim'
UEPY_GITHUB = 'https://github.com/deepdrive/UnrealEnginePython'
IS_CUSTOM_CI = 'CUSTOM_CI' in os.environ
UNREAL_DIR = os.path.join(HOME_DIR, 'UnrealEngine')
UNREAL_PLUGINS_DIR = os.path.join(UNREAL_DIR, 'Engine', 'Plugins')
UEPY_NAME = 'UnrealEnginePython'
UEPY_DIR = os.path.join(UNREAL_PLUGINS_DIR, UEPY_NAME)
LOCAL_PLUGINS_DIR = os.path.join(SIM_DIR, 'Plugins')
LOCAL_UEPY_DIR = os.path.join(LOCAL_PLUGINS_DIR, UEPY_NAME)
STAGE = os.environ.get('PKG_STAGE', 'all')   # requirements | uepy | package
SUBSTANCE_NAME = 'Substance'
SUBSTANCE_DIR = os.path.join(UNREAL_PLUGINS_DIR, 'Runtime', SUBSTANCE_NAME)
LOCAL_SUBSTANCE_DIR = os.path.join(LOCAL_PLUGINS_DIR, SUBSTANCE_NAME)
SUBSTANCE_URL = 'https://s3-us-west-1.amazonaws.com/deepdrive/substance/SubstanceUE4_Plugin_Public_4.21.0.31.zip'


cwd = os.getcwd()
os.chdir(SIM_DIR)  # Hack around os.getcwd() used for cli
from ue4cli import UnrealManagerFactory  # Included with ue4-full docker image

log.basicConfig(format='%(asctime)-15s - %(levelname)s - %(message)s', level=log.INFO)


def main():
    assert_in_ue4_docker()
    run_all = STAGE == 'all'
    log.info('Cleaning derived data (Intermediate, Binaries, etc...)')
    unreal_clean(SIM_DIR)
    if not run_all:
        log.info('PKG_STAGE set to "%s" *Only* running this stage' % STAGE)
    if run_all or STAGE == 'requirements':
        pypip.ensure(PYTHON_REQUIREMENTS)
        maybe_get_sources()
    if run_all or STAGE == 'substance':
        ensure_substance_sources()
        log.info('Ensured Substance plugin sources')
    if run_all or STAGE == 'substance_build':
        log.info('Checking for Substance binaries')
        ensure_substance_binaries()
        log.info('Ensured Substance binaries')
    if run_all or STAGE == 'uepy_src':
        ensure_uepy_sources()
        log.info('Ensured UEPy sources')
    if run_all or STAGE == 'uepy_build':
        ensure_latest_uepy_binaries()
        log.info('Ensured UEPy binaries')
    if run_all or STAGE == 'package':
        log.info('Packaging sim with ue4cli')
        package_sim()


def ensure_substance_binaries():
    import utils
    if os.name == 'nt':
        if utils.has_stuff(os.path.join(SUBSTANCE_DIR, 'Binaries', 'Win64')):
            log.info('Substance for Windows is already compiled')
            return
    if utils.has_stuff(os.path.join(SUBSTANCE_DIR, 'Binaries', 'Linux')):
        log.info('Substance plugin appears to have Binaries')
    else:
        log.info('No Substance binaries, compiling Substance')
        print()
        system(os.path.join(UNREAL_DIR, 'Setup.sh'))
        system(os.path.join(UNREAL_DIR, 'GenerateProjectFiles.sh'))
        system(os.path.join(UNREAL_DIR, 'make'))


def system(command):
    ret = os.system(command)
    if ret != 0:
        raise RuntimeError('The following command failed: %s' % command)


def maybe_get_sources():
    if IS_CUSTOM_CI:
        log.info('Setting up a clean source directory in %s' % SIM_DIR)
        get_clean_sim_sources()


def ensure_substance_sources():
    import utils

    if os.path.exists(SUBSTANCE_DIR):
        log.info('Found Substance Plugin sources')

    else:
        substance_tmp_dir = os.path.join(UNREAL_PLUGINS_DIR, 'Runtime', 'SubstanceTempDir')
        utils.download(SUBSTANCE_URL, substance_tmp_dir, warn_existing=False, overwrite=False)
        subfolder = glob(os.path.join(substance_tmp_dir, '*/'))[0]
        subfolder = os.path.join(subfolder, 'Plugins', 'Runtime', 'Substance')
        shutil.move(subfolder, SUBSTANCE_DIR)
        shutil.rmtree(substance_tmp_dir)


def compile_plugin_locally(local_dir, final_dir, name):
        is_git = is_git_repo(final_dir)

        def clean_local_dir():
            if is_git:
                log.info('Running git clean in %s...', local_dir)
                git_clean(local_dir)
            else:
                log.info('Cleaning derived data from %s...', local_dir)
                unreal_clean(local_dir)
        if os.path.exists(local_dir):
            raise RuntimeError('Local %s plugin exists - cannot move engine plugin' % name)
        shutil.move(final_dir, local_dir)
        log.info('Moved %s to %s', final_dir, local_dir)
        clean_local_dir()
        log.info('Compiling %s', name)
        build_exception = None
        try:
            build_sim()
        except Exception as e:
            log.error('Build failed, restoring sources to Engine plugins')
            clean_local_dir()
            build_exception = e
        finally:
            shutil.move(local_dir, final_dir)
            log.info('Moved %s to %s' % (local_dir, final_dir))
        if build_exception is not None:
            raise build_exception


def ensure_uepy_sources():
    if not os.path.exists(UEPY_DIR):
        git_clone(url=UEPY_GITHUB, path=UEPY_DIR)


def get_uepy_path():
    if os.path.exists(UEPY_DIR):
        if os.path.exists(LOCAL_UEPY_DIR):
            raise RuntimeError('Expect %s to be in either engine or project plugins, but found in both' % UEPY_NAME)
        else:
            uepy_path = UEPY_DIR
    elif os.path.exists(LOCAL_UEPY_DIR):
        uepy_path = LOCAL_UEPY_DIR
    else:
        return None
    return uepy_path

# DONE: Build a base docker image with UE4 4.21 and push it to a _private_ docker registry to comply with licensing
# So needs to build from scratch with ue4-docker

# DONE: docker run with volume mounted to local, then package

# DONE: if deepdrive-sim does not exist
# git clone deepdrive-sim --branch deepdrive_sim
# cd deepdrive-sim
# git clean, clean.sh and pull latest!

# DONE: if Engine/Plugins/UnrealEnginePython does not exist or new UEPy commits
# DONE: git clone https://github.com/deepdrive/UnrealEnginePython deepdrive-sim/Plugins
# ue4 build
# mv UnrealEnginePython Engine/Plugins
# DONE: ue4-package Development

# Upload the packaged binaries to s3

# Commit the docker container with our project and push to the private docker server on AWS.
# This will allow us to cache things how we want instead of relying on docker's line based cache


def package_sim():
    """
    Package sim in Development mode to get yummy command line tools but with Shipping performance
    """
    manager = UnrealManagerFactory.create()

    # Monkey patch build
    orig_build = manager.buildProject
    manager.buildProject = functools.partial(orig_build, dir=SIM_DIR)

    manager.packageProject(configuration='Development',
                           # TODO: Determine effect of these removed args and the added -allmaps arg
                           # extraArgs=['-nocompileeditor',
                           #            '-clean']
                           )

    # Restore original build
    manager.buildProject = orig_build


def unreal_clean(path):
    def rm(*paths):
        p = os.path.join(path, *paths)
        if os.path.exists(p):
            shutil.rmtree(p)
    rm('DerivedDataCache')
    rm('Intermediate')
    rm('Binaries')
    rm('Plugins', 'DeepDrivePlugin', 'Binaries')
    rm('Plugins', 'DeepDrivePlugin', 'Intermediate')


def ensure_latest_uepy_binaries():
    """
    UEPY does not compile as an engine plugin so we compile locally then move it to the engine's
    plugins to avoid recompiling later
    """
    import utils
    if UEPY_DIR is None:
        raise ValueError('Could not find UEPy')
    if github_repo_same_as_local(UEPY_DIR):
        log.info('UnrealEnginePython is up to date')
        if utils.has_stuff(os.path.join(UEPY_DIR, 'Binaries')):
            log.info('UEPy has binaries - assuming up-to-date')
            recompile = False
        else:
            log.info('No UEPy binaries, compiling...')
            recompile = True
    else:
        log.info('%s is not in sync with github. Grabbing latest and recompiling as a local plugin' % UEPY_NAME)
        log.info('Pulling latest into %s...' % UEPY_DIR)
        git_pull_latest(UEPY_DIR)
        if not github_repo_same_as_local(UEPY_DIR):
            raise RuntimeError('Not able to sync local and github repo. Do you have local commits that aren\'t pushed?')
        recompile = True

    if recompile:
        compile_plugin_locally(LOCAL_UEPY_DIR, UEPY_DIR, UEPY_NAME)


def build_sim():
    manager = UnrealManagerFactory.create()
    manager.buildProject(SIM_DIR, 'Development')


def assert_in_ue4_docker():
    if not is_docker():
        raise RuntimeError('Package is expected to be run within docker')
    log.info('Confirmed we are running in a docker container')
    if HOME_DIR != '/home/ue4':
        raise RuntimeError('Unexpected home directory, '
                           'expected /home/ue4 '
                           'got %s. Are you running with docker?' %
                           HOME_DIR)
    log.info('Confirmed that home directory is /home/ue4')


def git_pull_latest(repo_dir, remote='origin'):
    import git
    repo = git.Repo(repo_dir)
    repo.remotes[remote].pull(progress=custom_progress())


def get_clean_sim_sources():
    # For local testing or not using Jenkins / Travis
    if not os.path.exists(SIM_DIR):
        git_clone(url=SIM_GITHUB, path=SIM_DIR, branch=SIM_BRANCH)
    else:
        git_clean(SIM_DIR)
        git_pull_latest(SIM_DIR)


def github_repo_same_as_local(repo_dir, remote='origin'):
    import git
    from github import Github
    local_repo = git.Repo(repo_dir)
    remote_path = get_remote_path(local_repo, remote)
    remote_repo = Github().get_repo(remote_path)
    remote_commit = remote_repo.get_branch(local_repo.active_branch.name).commit.sha
    local_commit = str(local_repo.commit())
    same = remote_commit == local_commit
    return same


def get_remote_path(local_repo, remote='origin'):
    url = list(local_repo.remotes[remote].urls)[0]
    if url.startswith('https'):
        remote_path = '/'.join(url.split('/')[-2:])
    elif url.startswith('git@'):
        remote_path = url[url.find(':') + 1:]
    else:
        raise ValueError('Unexpected remote URL format: %s' % (url,))
    return remote_path


def git_clean(repo_dirMoved ):
    import git
    git.Repo(repo_dir).git.clean('-xdf')


def is_git_repo(repo_dir):
    import git
    try:
        git.Repo(repo_dir)
    except git.InvalidGitRepositoryError:
        return False
    else:
        return True


def custom_progress():
    from git import RemoteProgress

    class CustomProgress(RemoteProgress):
        def __init__(self):
            self.last_print_time = None
            super(CustomProgress, self).__init__()

        def update(self, op_code, cur_count, max_count=None, message=''):
            if message:
                now = time.time()
                if self.last_print_time is None or now - self.last_print_time > 5:
                    log.info(message)
                    self.last_print_time = now

    return CustomProgress()


def git_clone(url, path, branch='master'):
    import git
    log.info('Cloning into %s' % path)
    git.Repo.clone_from(url, path, branch=branch,
                        progress=custom_progress())


def is_docker():
    path = '/proc/self/cgroup'
    return (
        os.path.exists('/.dockerenv') or
        os.path.isfile(path) and any('docker' in line for line in open(path))
    )


if __name__ == '__main__':
    main()
