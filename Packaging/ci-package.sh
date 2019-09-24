#!/usr/bin/env bash

git clone --depth=1 --branch ${DEEPDRIVE_BRANCH} https://github.com/deepdrive/deepdrive-sim
cd deepdrive-sim
git checkout -qf ${DEEPDRIVE_COMMIT}

# https://stackoverflow.com/a/42982144/134077
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

python3 -u Packaging/package.py --upload
