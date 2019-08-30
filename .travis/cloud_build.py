import os
import sys

import time

import requests
from box import Box
from loguru import logger as log
from retry import retry


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
    handle_job_results(job_id)


def handle_job_results(job_id):
    job = wait_for_job_to_finish(job_id)
    if job.results.errors:
        log.error(f'Build finished with errors. Job details:\n'
                  f'{job.to_json(indent=2, default=str)}')
        exit_code = 1
    else:
        log.success(f'Build finished successfully. Job details:\n'
                    f'{job.to_json(indent=2, default=str)}')
        exit_code = 0
    if job.results.logs:
        log.info(f'Full job logs: '
                 f'{job.results.logs.to_json(indent=2, default=str)}')
    sys.exit(exit_code)


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
    handle_job_results(
        '2019-08-30_06-20-06PM_5bc353a24467e3e54c73bf6c4d82129b99363f7d')


if __name__ == '__main__':
    if 'test_handle_job_results' in sys.argv:
        test_handle_job_results()
    elif 'test_wait_for_job_to_finish' in sys.argv:
        test_wait_for_job_to_finish()
    else:
        main()
