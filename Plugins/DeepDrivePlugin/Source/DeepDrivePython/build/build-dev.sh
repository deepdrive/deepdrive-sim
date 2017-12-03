#!/usr/bin/env bash

set -euxo pipefail

ROOT=`git rev-parse --show-toplevel`


# Set patch version to git commit time
export DEEPDRIVE_VERSION=`python "${ROOT}"/Packaging/get_package_version.py`
export DEEPDRIVE_BRANCH=`git rev-parse --abbrev-ref HEAD`

echo "DEEPDRIVE_VERSION is $DEEPDRIVE_VERSION"
echo "DEEPDRIVE_BRANCH is DEEPDRIVE_BRANCH"

python setup.py install