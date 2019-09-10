import json
import os
import sys
import time
from datetime import datetime
from os.path import dirname, realpath, join

import requests
from box import Box
from github import Github
from loguru import logger as log
from retry import retry
from github import UnknownObjectException

import problem_constants.constants

"""
Here we run the build and integration tests on our own infrastructure so that
we can have full control over how things are run, but still keep track of things
in a standard way.
"""

DIR = dirname(realpath(__file__))
PACKAGE_DIR = join(dirname(DIR), 'Packaging')
sys.path.insert(0, PACKAGE_DIR)

from get_package_version import get_package_version


@log.catch(reraise=True)
def main():
    if os.environ.get('ENABLE_SIM_BUILD') != 'true':
        log.warning('Not building sim without ENABLE_SIM_BUILD=true')
        return

    build_id = os.environ.get('TRAVIS_BUILD_ID') or \
               generate_rand_alphanumeric(6)
    commit = os.environ['TRAVIS_COMMIT']
    branch = os.environ['TRAVIS_BRANCH']
    resp = requests.post('https://sim.deepdrive.io/build', json=dict(
        build_id=build_id,
        commit=commit,
        branch=branch,
    ))
    job_id = resp.json()['job_id']
    build_success = wait_for_build_result(job_id)
    if not build_success:
        raise RuntimeError('Cloud build failed')
    else:
        run_botleague_ci(branch)


@log.catch(reraise=True)
def run_botleague_ci(branch):
    # Send pull request to Botleague
    # TODO: with the new container and source commit for each problem we
    #  support, right now just domain randomization.
    #  The pull request will specify “test_only”: true (OR BE A DRAFT?)
    #  in problem.json if the pull request is not meant to be merged
    #  into botleague. I.e. It’s just for a dev branch.
    log.info('Sending pull requests to botleague for supported problems')
    github_token = os.environ['BOTLEAGUE_GITHUB_TOKEN']
    github_client = Github(github_token)
    repo = github_client.get_repo('deepdrive/deepdrive-sim')
    # Get our fork owner
    botleague_fork_owner = 'deepdrive'
    # github_client.get_user('someusername')  # or that for a user fork
    # NOTE: Fork on github was manually created
    botleague_repo = github_client.get_repo('botleague/botleague')
    botleague_fork = github_client.get_repo(
        f'{botleague_fork_owner}/botleague')
    for problem in problem_constants.constants.SUPPORTED_PROBLEMS:

        # TODO: This is returning stale data (also get_branches)
        #   May need to clone the repo!
        hash_to_branch_from = get_head('botleague/botleague', github_token)

        sim_version = get_package_version()
        botleague_branch_name = f'deepdrive_{get_package_version()}_' \
            f'{generate_rand_alphanumeric(3)}'
        botleague_fork.create_git_ref(
            ref=f'refs/heads/{botleague_branch_name}',
            sha=hash_to_branch_from)
        problem_json_path = f'problems/deepdrive/{problem}/problem.json'
        problem_def, problem_sha = get_file_from_github(
            botleague_fork,
            problem_json_path,
            ref=botleague_branch_name
        )
        if dbox(problem_def).version == sim_version:
            date_str = datetime.utcnow().strftime('%Y-%m-%d_%I-%M-%S%p')
            problem_def.rerun = date_str
        problem_def.version = sim_version
        message = 'Auto generated commit for deepdrive-sim CI'

        # Add a newline before comments
        content = problem_def.to_json(indent=2) \
            .replace('\n  "$comment-', '\n\n  "$comment-')

        botleague_fork.update_file(problem_json_path, message,
                                   content=content,
                                   sha=problem_sha,
                                   branch=botleague_branch_name)
        pull = Box(
            title=f'CI trigger for {botleague_branch_name}',
            body=f'',
            head=f'{botleague_fork_owner}:{botleague_branch_name}',
            base='master',)
        if branch not in ['master', 'v3_stable']:
        # if branch not in ['master']:
            pull.draft = True

        pull_resp = create_pull_request(
            pull, repo_full_name='botleague/botleague',
            token=github_token)
    # TODO: Poll botleague problem run statuses with pull # to collect results?
    # TODO: For each problem, check that results are within range


def wait_for_build_result(job_id) -> bool:
    job = wait_for_job_to_finish(job_id)
    if job.results.errors:
        log.error(f'Build finished with errors. Job details:\n'
                  f'{job.to_json(indent=2, default=str)}')
        ret = False
    else:
        log.success(f'Build finished successfully. Job details:\n'
                    f'{job.to_json(indent=2, default=str)}')
        ret = True
    if job.results.logs:
        log.info(f'Full job logs: '
                 f'{job.results.logs.to_json(indent=2, default=str)}')

    return ret

def wait_for_job_to_finish(job_id) -> Box:
    last_log_time = None
    log.info(f'Waiting for sim build {job_id} to complete...')
    while True:
        status_resp = get_job_status(job_id)
        job_status = dbox(status_resp.json())
        if job_status.status == 'finished':
            return job_status
        elif not last_log_time or time.time() - last_log_time > 5:
            print('.', end='', flush=True)
            last_log_time = time.time()
        time.sleep(1)


@log.catch
@retry(tries=5, jitter=(0, 1), logger=log)
def get_job_status(job_id):
    status_resp = requests.post('https://sim.deepdrive.io/job/status',
                                json={'job_id': job_id})
    if not status_resp.ok:
        raise RuntimeError('Error getting job status')
    return status_resp


def generate_rand_alphanumeric(num_chars: int) -> str:
    from secrets import choice
    import string
    alphabet = string.ascii_lowercase + string.digits
    ret = ''.join(choice(alphabet) for _ in range(num_chars))
    return ret


def dbox(obj):
    return Box(obj, default_box=True)


def test_wait_for_job_to_finish():
    wait_for_job_to_finish(
        '2019-08-30_10-24-31PM_fc1a2e6dc4744697b1b68728063c6990f5633afe')


def test_handle_job_results():
    wait_for_build_result(
        '2019-08-30_06-20-06PM_5bc353a24467e3e54c73bf6c4d82129b99363f7d')


def get_file_from_github(repo, filename, ref=None):
    """@:param filename: relative path to file in repo"""
    try:
        args = [filename]
        if ref is not None:
            args.append(ref)
        contents = repo.get_contents(*args)
        ret_sha = contents.sha
        content_str = contents.decoded_content.decode('utf-8')
    except UnknownObjectException:
        log.error(f'Unable to find {filename} in {repo.html_url}')
        content_str = ''
    ret_content = get_str_or_box(content_str, filename)
    return ret_content, ret_sha


def get_str_or_box(content_str, filename):
    if filename.endswith('.json') and content_str:
        ret = Box(json.loads(content_str))
    else:
        ret = content_str
    return ret

def create_pull_request(pull: Box, repo_full_name: str, token: str) -> \
        requests.Response:
    """Doing this manually until
    https://github.com/PyGithub/PyGithub/issues/1213 is resolved
    """
    headers = dict(
        Accept='application/vnd.github.shadow-cat-preview+json',
        Authorization=f'token {token}'
    )
    resp = requests.post(
        f'https://api.github.com/repos/{repo_full_name}/pulls',
        json=pull.to_dict(),
        headers=headers)
    log.info(f'Pull request response {resp.json(indent=2)}')
    return resp


def get_head(full_repo_name: str, token: str, branch: str = 'master'):
    headers = dict(Authorization=f'token {token}')
    resp = requests.get(
        f'https://api.github.com/repos/'
        f'{full_repo_name}/git/refs/heads/{branch}',
        headers=headers)
    ret = resp.json()['object']['sha']
    return ret


def play():
    get_head('botleague/botleague', os.environ['BOTLEAGUE_GITHUB_TOKEN'])
    # github_client = Github(os.environ['BOTLEAGUE_GITHUB_TOKEN'])
    # botleague_repo = github_client.get_repo(f'deepdrive/botleague')
    # problem_json_path = f'problems/deepdrive/domain_randomization/problem.json'
    # problem_def, problem_sha = get_file_from_github(botleague_repo,
    #                                                 problem_json_path)

if __name__ == '__main__':
    if 'test_handle_job_results' in sys.argv:
        test_handle_job_results()
    elif 'test_wait_for_job_to_finish' in sys.argv:
        test_wait_for_job_to_finish()
    elif 'run_botleague_ci' in sys.argv:
        run_botleague_ci('v3_stable')
    elif 'play' in sys.argv:
        play()
    else:
        main()
