import json
import os
import sys
from typing import List, Tuple
import requests

import time
from datetime import datetime
from os.path import dirname, realpath, join

from botleague_helpers.ci import build_and_run_botleague_ci, run_botleague_ci, \
    dbox
from botleague_helpers.utils import box2json
from box import Box

import problem_constants.constants

from loguru import logger as log
# from logs import log

"""
Here we run the build and integration tests on our own infrastructure so that
we can have full control over how the build is run, but still keep track builds
in a standard way with a nice, free, hosted, UI.
"""

DIR = dirname(realpath(__file__))
PACKAGE_DIR = join(dirname(DIR), 'Packaging')
sys.path.insert(0, PACKAGE_DIR)

from get_package_version import get_package_version
from package import copy_release_candidate_to_release, upload_s3_str


@log.catch(reraise=True)
def main():
    if os.environ.get('ENABLE_SIM_BUILD') != 'true':
        log.warning('Not building sim without ENABLE_SIM_BUILD=true')
        return
    build_and_run_botleague_ci(
        build_url='https://sim.deepdrive.io/build',
        run_botleague_ci_wrapper_fn=run_botleague_ci_for_sim_build)


def promote_python_bindings(sim_version, commit, job):
    # Check to see that the bindings have been uploaded to pypi (wait for them)
    url = f'https://deepdrive.s3-us-west-1.amazonaws.com/' \
        f'unvalidated-bindings-versions/{sim_version}'
    log.info('Waiting for Travis to record python bindings')
    start = time.time()

    while not requests.head(url).ok:
        time.sleep(1)
        if time.time() - start > (60 * 5):
            raise RuntimeError('Timeout waiting for bindings to record, '
                               'check Travis')
    log.success('Python bindings uploaded')
    content = box2json(Box(commit=commit, build_job=job, version=sim_version))
    upload_s3_str(content, f'validated-bindings-versions/{sim_version}')


def run_botleague_ci_for_sim_build(branch, commit, job):
    build_results = Box.from_json(job.results.json_results_from_logs)
    sim_version = get_package_version()

    def set_version(problem_def, version):
        if dbox(problem_def).version == version:
            date_str = datetime.utcnow().strftime('%Y-%m-%d_%I-%M-%S%p')
            problem_def.rerun = date_str
        problem_def.version = version

    pr_message = 'Auto generated commit for deepdrive-sim CI'
    passed_ci = run_botleague_ci(
        branch=branch,
        version=sim_version,
        set_version_fn=set_version,
        pr_message=pr_message,
        sim_url=build_results.gcs_url,
        supported_problems=problem_constants.constants.SUPPORTED_PROBLEMS)

    if passed_ci and branch in ['master']:  # TODO: Use botleague helpers release branches
        # Copy sim/release_candidates/ to sim/
        copy_release_candidate_to_release()
        promote_python_bindings(sim_version, commit, job)

if __name__ == '__main__':
    if '--promote-bindings-version' in sys.argv:
        promote_python_bindings('3.0.20191010224955', commit='asdf',
                                job=Box(a=1))
    else:
        main()
