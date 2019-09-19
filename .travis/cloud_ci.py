import json
import os
import sys
from typing import List

import time
from datetime import datetime
from os.path import dirname, realpath, join

import requests
from box import Box, BoxList
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
from package import copy_release_candidate_to_release

DEFAULT_BOTLEAGUE_LIAISON_HOST = 'https://liaison.botleague.io'
BOTLEAGUE_LIAISON_HOST = os.environ.get('BOTLEAGUE_LIAISON_HOST') or \
                         DEFAULT_BOTLEAGUE_LIAISON_HOST


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
        passed_ci = run_botleague_ci(branch)
        if passed_ci:
            # Copy sim/release_candidates/ to sim/
            copy_release_candidate_to_release()


@log.catch(reraise=True)
def run_botleague_ci(branch) -> bool:
    # Send pull request to Botleague
    log.info('Sending pull requests to botleague for supported problems')
    github_token = os.environ['BOTLEAGUE_GITHUB_TOKEN']
    github_client = Github(github_token)
    # Get our fork owner
    botleague_fork_owner = 'deepdrive'
    # github_client.get_user('someusername')  # or that for a user fork
    # NOTE: Fork on github was manually created
    botleague_fork = github_client.get_repo(
        f'{botleague_fork_owner}/botleague')
    problem_cis = BoxList()
    for problem in problem_constants.constants.SUPPORTED_PROBLEMS:
        hash_to_branch_from = get_head('botleague/botleague', github_token)

        sim_version = get_package_version()
        botleague_branch_name = f'deepdrive_{get_package_version()}_' \
            f'id-{generate_rand_alphanumeric(3)}'
        fork_ref = botleague_fork.create_git_ref(
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

        update_resp = botleague_fork.update_file(problem_json_path, message,
                                   content=content,
                                   sha=problem_sha,
                                   branch=botleague_branch_name)
        pull = Box(
            title=f'CI trigger for {botleague_branch_name}',
            body=f'',
            head=f'{botleague_fork_owner}:{botleague_branch_name}',
            base='master',)
        if BOTLEAGUE_LIAISON_HOST != DEFAULT_BOTLEAGUE_LIAISON_HOST:
            pull.body = Box(
                botleague_liaison_host=BOTLEAGUE_LIAISON_HOST).to_json()

        # if branch not in ['master']:
        if branch not in ['master', 'v3_stable']:
            pull.draft = True

        pull_resp = create_pull_request(
            pull, repo_full_name='botleague/botleague',
            token=github_token)

        head_sha = Box(update_resp).commit.sha
        problem_cis.append(Box(pr_number=pull_resp.json()['number'],
                               commit=head_sha))
    problem_cis = wait_for_problem_cis(problem_cis)
    if all(p.status == 'passed' for p in problem_cis):
        log.success(f'Problem ci\'s passed! Problem cis were: '
                 f'{problem_cis.to_json(indent=2, default=str)}')
        return True
    else:
        raise RuntimeError(f'Problem ci\'s failed! Problem cis were: '
                 f'{problem_cis.to_json(indent=2, default=str)}')


def wait_for_problem_cis(problem_cis: BoxList) -> BoxList:
    log.info(f'Waiting for problem cis {problem_cis.to_json()} to complete...')
    def problem_cis_complete():
        for pci in problem_cis:
            status_resp = get_problem_ci_status(pci.pr_number, pci.commit)
            status_resp = dbox(status_resp.json())
            pci.status = status_resp.status
            if status_resp.status == 'pending':
                return None
            elif status_resp.status == 'not-found':
                raise RuntimeError(f'Problem CI not found. Looking for: '
                                   f'{box2json(pci)}')
        return problem_cis
    return wait_for_fn(problem_cis_complete)


def wait_for_fn(fn: callable):
    last_log_time = None
    while True:
        ret = fn()
        if ret is not None:
            return ret
        elif not last_log_time or time.time() - last_log_time > 5:
            print('.', end='', flush=True)
            last_log_time = time.time()
        time.sleep(1)


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
def get_problem_ci_status(pr_number: int, commit: str):
    status_resp = requests.post(f'{BOTLEAGUE_LIAISON_HOST}/problem_ci_status',
                                json=dict(commit=commit, pr_number=pr_number))
    if not status_resp.ok:
        raise RuntimeError('Error getting job status')
    return status_resp


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
    log.info(f'Created pull request #{resp.json()["number"]} on '
             f'{repo_full_name}')
    return resp


def get_head(full_repo_name: str, token: str, branch: str = 'master'):
    headers = dict(Authorization=f'token {token}')
    resp = requests.get(
        f'https://api.github.com/repos/'
        f'{full_repo_name}/git/refs/heads/{branch}',
        headers=headers)
    ret = resp.json()['object']['sha']
    return ret


def box2json(box: Box):
    return box.to_json(indent=2, default=str)


def play():
    get_head('botleague/botleague', os.environ['BOTLEAGUE_GITHUB_TOKEN'])
    # github_client = Github(os.environ['BOTLEAGUE_GITHUB_TOKEN'])
    # botleague_repo = github_client.get_repo(f'deepdrive/botleague')
    # problem_json_path = f'problems/deepdrive/domain_randomization/problem.json'
    # problem_def, problem_sha = get_file_from_github(botleague_repo,
    #                                                 problem_json_path)


def test_wait_for_problem_cis():
    ret = wait_for_problem_cis(BoxList([
        Box(pr_number=96, commit='4a6a67702931920ee9a6813247d09d88b433d7b0')]))
    return ret


if __name__ == '__main__':
    if 'test_handle_job_results' in sys.argv:
        test_handle_job_results()
    elif 'test_wait_for_problem_cis' in sys.argv:
        test_wait_for_problem_cis()
    elif 'test_wait_for_job_to_finish' in sys.argv:
        test_wait_for_job_to_finish()
    elif 'run_botleague_ci' in sys.argv:
        run_botleague_ci('v3_stable')
    elif 'play' in sys.argv:
        play()
    else:
        main()
