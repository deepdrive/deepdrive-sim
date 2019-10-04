import json
import os
import sys
from typing import List, Tuple

import time
from datetime import datetime
from os.path import dirname, realpath, join

from botleague_helpers.ci import build_and_run_botleague_ci, run_botleague_ci, \
    dbox
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
from package import copy_release_candidate_to_release


@log.catch(reraise=True)
def main():
    if os.environ.get('ENABLE_SIM_BUILD') != 'true':
        log.warning('Not building sim without ENABLE_SIM_BUILD=true')
        return
    build_and_run_botleague_ci(
        build_url='https://sim.deepdrive.io/build',
        run_botleague_ci_wrapper_fn=run_botleague_ci_for_sim_build)


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

    if passed_ci:
        # Copy sim/release_candidates/ to sim/
        copy_release_candidate_to_release()
        # TODO: Promote python bindings to PyPi default
