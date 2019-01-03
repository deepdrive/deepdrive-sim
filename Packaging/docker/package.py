# Goal: Continuous integration packaging in docker, upload to s3, build env image, test benchmark in image

# Usage: See Dockerfile

import os
import shutil
import time

from ue4cli import UnrealManagerFactory  # Included with ue4-full docker image

import pypip

PYTHON_REQUIREMENTS = [
    {'module': 'git', 'pip': 'gitpython'},
    {'module': 'github', 'pip': 'PyGithub'},
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


def main():
    assert_in_ue4_docker()
    pypip.ensure(PYTHON_REQUIREMENTS)
    ensure_repo_clean()
    ensure_uepy()
    package_sim()


def ensure_repo_clean():
    if IS_CUSTOM_CI:
        get_clean_sim_sources()


def ensure_uepy():
    local_path = os.path.join(LOCAL_PLUGINS_DIR, UEPY_NAME)
    if os.path.exists(UEPY_DIR):
        if os.path.exists(local_path):
            raise RuntimeError('Expect %s to be in either engine or project plugins, but found in both' % UEPY_NAME)
    else:
        git_clone(url=UEPY_GITHUB, path=LOCAL_PLUGINS_DIR)
    ensure_latest_uepy_binaries()

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
    if github_repo_is_stale(UEPY_DIR):
        print('%s is stale. Grabbing latest and recompiling as local plugin', UEPY_NAME)
        shutil.move(UEPY_DIR, LOCAL_UEPY_DIR)
        git_clean(LOCAL_UEPY_DIR)
        git_pull_latest(LOCAL_UEPY_DIR)
        build_sim()
        shutil.move(LOCAL_UEPY_DIR, UEPY_DIR)


def build_sim():
    manager = UnrealManagerFactory.create()
    manager.buildProject(SIM_DIR, 'Development')


def assert_in_ue4_docker():
    if not is_docker():
        raise RuntimeError('Package is expected to be run within docker')

    if HOME_DIR != '/home/ue4':
        raise RuntimeError('Unexpected home directory, '
                           'expected /home/ue4 '
                           'got %s. Are you running with docker?' %
                           HOME_DIR)


def git_pull_latest(repo_dir, remote='origin'):
    import git
    repo = git.Repo(repo_dir)
    repo.remotes[remote].pull(progress=minutes_remaining_progress())


def get_clean_sim_sources():
    # For local testing or not using Jenkins / Travis
    if not os.path.exists(SIM_DIR):
        git_clone(url=SIM_GITHUB, path=SIM_DIR, branch=SIM_BRANCH)
        # TODO: # cd deepdrive-sim
        # TODO: git clean, clean.sh and pull latest


def github_repo_is_stale(repo_dir, remote='origin'):
    import git
    from github import Github
    local_repo = git.Repo(repo_dir)
    url = list(local_repo.remotes[remote].urls)[0]
    remote_repo = Github().get_repo(url[url.find(':') + 1:])
    remote_commit = remote_repo.get_branch(local_repo.active_branch.name).commit.sha
    local_commit = str(local_repo.commit())
    stale = remote_commit != local_commit
    return stale


def git_clean(repo_dir):
    import git
    git.Repo(repo_dir).git.clean('-xdf')


def minutes_remaining_progress():
    from git import RemoteProgress

    class MinutesRemainingProgress(RemoteProgress):
        def __init__(self):
            self.percent = 0
            self.last_print_time = None
            self.start = time.time()
            super(MinutesRemainingProgress, self).__init__()

        def update(self, op_code, cur_count, max_count=None, message=''):
            if message:
                self.print_minutes_remaining(cur_count, max_count, message)

        def print_minutes_remaining(self, cur_count, max_count, message):
            frac = cur_count / (max_count or 100.0)
            self.percent = frac * 100
            now = time.time()
            if self.last_print_time is None or now - self.last_print_time > 5:
                mins_remaining = (now - self.start) / frac / 60
                msg = '%0.1f%% finished: %s' % (self.percent, message)
                if mins_remaining >= 2:
                    msg = '%s ~ %d minutes remaining' % (msg, mins_remaining)
                print(msg)
                self.last_print_time = now

    return MinutesRemainingProgress()


def git_clone(url, path, branch='master'):
    import git
    print('Cloning into %s' % path)
    git.Repo.clone_from(url, path, branch=branch,
                        progress=minutes_remaining_progress())


def is_docker():
    path = '/proc/self/cgroup'
    return (
        os.path.exists('/.dockerenv') or
        os.path.isfile(path) and any('docker' in line for line in open(path))
    )


if __name__ == '__main__':
    main()
