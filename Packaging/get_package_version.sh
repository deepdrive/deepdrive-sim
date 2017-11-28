#!/usr/bin/env bash

set -euo > /dev/null

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

commit_time=`git log -1 --format=%ci`
patch_version=`python -u ${DIR}/get_patch_timestamp.py "$commit_time"`
echo `cat ${DIR}/VERSION`."$patch_version"