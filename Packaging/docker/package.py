# Goal: Continuous integration packaging in docker, upload to s3, build env image, test benchmark in image

# Usage: See Dockerfile

# Dev

# If you're debugging python and want to commit filesystem changes before some exception you can do this
# to commit the latest container
# docker commit `docker ps --latest --format "{{.ID}}"` deepdriveio/deepdrive-ue4

import os
import shutil
import time

from ue4cli import UnrealManagerFactory  # Included with ue4-full docker image

import pypip

PYTHON_REQUIREMENTS = [
    {'module': 'git', 'pip': 'gitpython'},
    {'module': 'github', 'pip': 'PyGithub'},
    'clint',
    'requests',
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
SUBSTANCE_URL = 'https://s3-us-west-1.amazonaws.com/deepdrive/substance/SubstanceUE4_Plugin_Public_4.21.0.31.zip'


def main():
    assert_in_ue4_docker()
    run_all = STAGE == 'all'
    if run_all or STAGE == 'requirements':
        pypip.ensure(PYTHON_REQUIREMENTS)
        ensure_substance_plugin()
        maybe_get_sources()
    if run_all or STAGE == 'uepy_src':
        ensure_uepy_sources()
    if run_all or STAGE == 'uepy_bin':
        ensure_latest_uepy_binaries()
    if run_all or STAGE == 'package':
        package_sim()
    else:
        raise RuntimeError('PKG_STAGE not recognized')


def maybe_get_sources():
    if IS_CUSTOM_CI:
        print('Setting up a clean source directory in %s' % SIM_DIR)
        get_clean_sim_sources()


def ensure_substance_plugin():
    from downloader import download
    substance_dir = os.path.join(UNREAL_PLUGINS_DIR, 'Runtime', 'Substance')
    download(SUBSTANCE_URL, substance_dir, warn_existing=False, overwrite=False)


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
    cwd = os.getcwd()
    manager = UnrealManagerFactory.create()
    os.chdir(SIM_DIR)
    manager.packageProject(configuration='Development')
    os.chdir(cwd)


def ensure_latest_uepy_binaries():
    """
    UEPY does not compile as an engine plugin so we compile locally then move it to the engine's
    plugins to avoid recompiling later
    """
    if UEPY_DIR is None:
        raise ValueError('Could not find UEPy')
    if github_repo_same_as_local(UEPY_DIR):
        print('UnrealEnginePython is up to date')
        if os.path.exists(os.path.join(UEPY_DIR, 'Binaries')):
            print('UEPy has binaries - assuming up-to-date')
            recompile = False
        else:
            print('No UEPy binaries, compiling...')
            recompile = True
    else:
        print('%s is not in sync with github. Grabbing latest and recompiling as a local plugin' % UEPY_NAME)
        print('Pulling latest into %s...' % UEPY_DIR)
        git_pull_latest(UEPY_DIR)
        if not github_repo_same_as_local(UEPY_DIR):
            raise RuntimeError('Not able to sync local and github repo. Do you have local commits that aren\'t pushed?')
        recompile = True

    if recompile:
        if os.path.exists(LOCAL_UEPY_DIR):
            raise RuntimeError('Local UnrealEnginePython plugin exists, moving')
        shutil.move(UEPY_DIR, LOCAL_UEPY_DIR)
        print('Moved %s to %s' % (UEPY_DIR, LOCAL_UEPY_DIR))
        print('Running git clean in %s...' % LOCAL_UEPY_DIR)
        git_clean(LOCAL_UEPY_DIR)
        print('Building project to create working UnrealEnginePython binaries')
        build_sim()
        shutil.move(LOCAL_UEPY_DIR, UEPY_DIR)
        print('Moved %s to %s' % (LOCAL_UEPY_DIR, UEPY_DIR))


def build_sim():
    manager = UnrealManagerFactory.create()
    manager.buildProject(SIM_DIR, 'Development')


def assert_in_ue4_docker():
    if not is_docker():
        raise RuntimeError('Package is expected to be run within docker')
    print('Confirmed we are running in a docker container')
    if HOME_DIR != '/home/ue4':
        raise RuntimeError('Unexpected home directory, '
                           'expected /home/ue4 '
                           'got %s. Are you running with docker?' %
                           HOME_DIR)
    print('Confirmed that home directory is /home/ue4')


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


def git_clean(repo_dir):
    import git
    git.Repo(repo_dir).git.clean('-xdf')


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
                    print(message)
                    self.last_print_time = now

    return CustomProgress()


def git_clone(url, path, branch='master'):
    import git
    print('Cloning into %s' % path)
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
