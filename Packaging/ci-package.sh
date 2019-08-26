#!/usr/bin/env bash

git clone --depth=1 --branch ${DEEPDRIVE_BRANCH} https://github.com/deepdrive/deepdrive-sim
cd deepdrive-sim
git checkout -qf ${DEEPDRIVE_COMMIT}
python3 -u Packaging/package.py
