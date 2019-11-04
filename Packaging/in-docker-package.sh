#!/usr/bin/env bash

set -e


DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

pip3 install -r ${DIR}/requirements-lock.txt
python3 ${DIR}/package.py --upload
